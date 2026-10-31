"""OpenRouter-backed embeddings and token accounting.

Follows the `ragbench.llm` pattern: an OpenAI-compatible client pointed at the
OpenRouter endpoint. Embeddings are always remote — no local models, no torch.
"""

from __future__ import annotations

import logging
from typing import Any

import openai
import tiktoken
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from zotero_rag.config import Config
from zotero_rag.errors import RejectedCredentialError

logger = logging.getLogger(__name__)

# OpenAI embedding models use cl100k_base; close enough for reporting on others.
_ENCODING = tiktoken.get_encoding("cl100k_base")

#: Texts per embedding request. LangChain's default of 1000 is OpenAI's limit, but
#: OpenRouter fronts providers with lower ones: Google's Vertex endpoint rejects
#: more than 250, so one long paper's chunks failed in a single request. A model
#: in the `embedding_models` catalogue can set its own `batch_size`.
DEFAULT_BATCH_SIZE = 100

#: Retries of a failed request. The OpenAI client backs off exponentially and
#: honours a provider's `Retry-After`, but LangChain's default of 2 gives up after
#: about 1.5s, which a rate-limited provider (Mistral, the free tiers) outlasts.
#: Six is roughly 25s of backoff before an attachment is skipped.
MAX_RETRIES = 6


def count_tokens(texts: list[str]) -> int:
    """Token count for a batch of texts, used for the `-v` usage report."""
    return sum(len(_ENCODING.encode(text)) for text in texts)


class OpenRouterEmbeddings(OpenAIEmbeddings):
    """`OpenAIEmbeddings` that names the rejected key instead of a bare 401.

    Without this a bad key reached the user as "unexpected error: Error code: 401
    - User not found", which does not say whether Zotero or OpenRouter refused it.
    """

    key_env: str = "OPENROUTER_API_KEY"

    def embed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs: Any
    ) -> list[list[float]]:
        try:
            return super().embed_documents(texts, chunk_size, **kwargs)
        except openai.AuthenticationError as exc:
            raise self._rejected(exc) from exc

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        try:
            return super().embed_query(text, **kwargs)
        except openai.AuthenticationError as exc:
            raise self._rejected(exc) from exc

    def _rejected(self, exc: openai.AuthenticationError) -> RejectedCredentialError:
        return RejectedCredentialError("OpenRouter", self.key_env, exc.message)


def build_embeddings(config: Config) -> OpenAIEmbeddings:
    """Build the embedding client. Requires an OpenRouter key."""
    config.global_.require_openrouter()
    spec = config.embedding_model_spec(config.embedding_model)
    return OpenRouterEmbeddings(
        key_env=config.global_.api_key_env,
        model=config.embedding_model,
        # LangChain's name for texts per request, not our characters per chunk.
        chunk_size=spec.batch_size if spec and spec.batch_size else DEFAULT_BATCH_SIZE,
        max_retries=MAX_RETRIES,
        # Canonical field names; `base_url`/`api_key` are their aliases.
        openai_api_base=config.global_.endpoint,
        openai_api_key=SecretStr(config.global_.api_key),
        # OpenRouter slugs such as "openai/text-embedding-3-small" are not in
        # tiktoken's model registry, so LangChain's token-based pre-chunking
        # cannot resolve them; our own splitter already bounds chunk size.
        check_embedding_ctx_length=False,
    )
