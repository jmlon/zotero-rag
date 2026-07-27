"""OpenRouter-backed embeddings and token accounting.

Follows the `ragbench.llm` pattern: an OpenAI-compatible client pointed at the
OpenRouter endpoint. Embeddings are always remote — no local models, no torch.
"""

from __future__ import annotations

import logging

import tiktoken
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from zotero_rag.config import Config

logger = logging.getLogger(__name__)

# OpenAI embedding models use cl100k_base; close enough for reporting on others.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(texts: list[str]) -> int:
    """Token count for a batch of texts, used for the `-v` usage report."""
    return sum(len(_ENCODING.encode(text)) for text in texts)


def build_embeddings(config: Config) -> OpenAIEmbeddings:
    """Build the embedding client. Requires an OpenRouter key."""
    config.global_.require_openrouter()
    return OpenAIEmbeddings(
        model=config.embedding_model,
        # Canonical field names; `base_url`/`api_key` are their aliases.
        openai_api_base=config.global_.endpoint,
        openai_api_key=SecretStr(config.global_.api_key),
        # OpenRouter slugs such as "openai/text-embedding-3-small" are not in
        # tiktoken's model registry, so LangChain's token-based pre-chunking
        # cannot resolve them; our own splitter already bounds chunk size.
        check_embedding_ctx_length=False,
    )
