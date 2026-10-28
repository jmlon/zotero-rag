"""OpenRouter-backed chat model and prompt/completion accounting.

Mirrors `embedding.build_embeddings`: an OpenAI-compatible client pointed at the
OpenRouter endpoint, with the credential resolved from the environment.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from zotero_rag.config import Config
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
