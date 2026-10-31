"""The embedding client's error reporting."""

from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from zotero_rag.embedding import OpenRouterEmbeddings
from zotero_rag.errors import RejectedCredentialError


@pytest.fixture
def embeddings() -> OpenRouterEmbeddings:
    """A client whose every request gets OpenRouter's answer to a revoked key."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            401, json={"error": {"message": "User not found.", "code": 401}}
        )
    )
    return OpenRouterEmbeddings(
        key_env="MY_OR_KEY",
        model="openai/text-embedding-3-small",
        openai_api_base="https://openrouter.example/api/v1",
        openai_api_key=SecretStr("sk-or-revoked"),
        http_client=httpx.Client(transport=transport),
        max_retries=0,
        check_embedding_ctx_length=False,
    )


def test_a_rejected_key_names_openrouter_and_the_variable(
    embeddings: OpenRouterEmbeddings,
) -> None:
    with pytest.raises(
        RejectedCredentialError, match="OpenRouter rejected the API key in MY_OR_KEY"
    ):
        embeddings.embed_documents(["some text"])


def test_a_rejected_key_is_named_for_queries_too(embeddings: OpenRouterEmbeddings) -> None:
    with pytest.raises(RejectedCredentialError, match="User not found"):
        embeddings.embed_query("a question")
