"""OpenRouter-backed chat model and prompt/completion accounting.

Mirrors `embedding.build_embeddings`: an OpenAI-compatible client pointed at the
OpenRouter endpoint, with the credential resolved from the environment.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from zotero_rag.config import Config
from zotero_rag.embedding import MAX_RETRIES
from zotero_rag.errors import ZoteroRagError

logger = logging.getLogger(__name__)


class LLMError(ZoteroRagError):
    """The chat completion call failed."""


def build_chat(config: Config) -> ChatOpenAI:
    """Build the chat client. Requires an OpenRouter key."""
    config.global_.require_openrouter()
    return ChatOpenAI(
        model=config.llm_model,
        temperature=config.index.temperature,
        # Comparisons answer from several threads at once; ride out a 429 instead
        # of losing the question to it.
        max_retries=MAX_RETRIES,
        # Canonical field names; `base_url`/`api_key` are their aliases.
        openai_api_base=config.global_.endpoint,
        openai_api_key=SecretStr(config.global_.api_key),
        # Ask for token usage on the final chunk of a stream; `invoke` reports it
        # regardless. Without this a streamed answer would report zero tokens.
        stream_usage=True,
    )


def usage_from_message(message: AIMessage) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) reported by the provider, 0 if absent.

    Usage metadata is advisory — some OpenRouter providers omit it — so a missing
    count is reported as zero rather than failing the query.
    """
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def cost_from_message(message: AIMessage) -> float | None:
    """What the provider says the call cost, in USD, or `None` if it did not say.

    OpenRouter reports it in `usage.cost`, which LangChain passes through in the
    response metadata. The actual charge beats a price list kept by hand: it
    follows price changes and the provider OpenRouter routed to.
    """
    usage = (message.response_metadata or {}).get("token_usage") or {}
    cost = usage.get("cost") if isinstance(usage, dict) else None
    return float(cost) if isinstance(cost, int | float) else None


def cached_tokens_from_message(message: AIMessage) -> int:
    """Input tokens the provider served from its prompt cache, 0 if it did not say."""
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    details = usage.get("input_token_details") or {}
    return int(details.get("cache_read", 0) or 0)


class ModelPrice(BaseModel):
    """A chat model's list price, USD per token, as OpenRouter publishes it."""

    prompt: float
    completion: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """The call's price with nothing served from the prompt cache."""
        return prompt_tokens * self.prompt + completion_tokens * self.completion


def fetch_price(endpoint: str, model: str) -> ModelPrice | None:
    """`model`'s list price from OpenRouter's public model list, or `None`.

    The charge a run reports depends on what the provider still had cached from
    the run before: two identical runs minutes apart differed by a factor of
    four. A query from a user is rarely a repeat, so the list price, with no
    cache discount, is the cost to compare configurations by.
    """
    try:
        response = httpx.get(f"{endpoint.rstrip('/')}/models", timeout=30.0)
        response.raise_for_status()
        listed = {entry.get("id"): entry.get("pricing") or {} for entry in response.json()["data"]}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("could not fetch model prices: %s", exc)
        return None
    pricing: dict[str, Any] = listed.get(model) or {}
    try:
        return ModelPrice(prompt=float(pricing["prompt"]), completion=float(pricing["completion"]))
    except (KeyError, ValueError):
        logger.warning("no list price for %s", model)
        return None


def message_text(message: AIMessage) -> str:
    """The message content as plain text.

    `AIMessage.content` is typed as a string *or* a list of content blocks; the
    list form appears with multimodal providers, so flatten it rather than
    stringifying a Python list into the answer.
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
