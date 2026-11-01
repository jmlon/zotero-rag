"""Reranking retrieved candidates before they reach the answering model (§11.4.3).

Retrieval ranks by *similarity*; a reranker ranks by *relevance to this question*.
A cross-encoder reads the question and each passage together, so it can tell that a
passage merely on the same subject is not the one that answers what was asked — a
judgement no embedding comparison can make, because the embedding was computed
without knowing the question.

It composes with everything upstream. Reranking consumes whatever the configured
strategy produced — vector or hybrid — and runs *before* windowing, which is the only
order that makes sense: a window's neighbours are context for a hit, not candidates
in their own right, so they must be added to the selection the reranker has already
made rather than competing in it.

A hosted reranker is used rather than a local cross-encoder because a local one
would mean `sentence-transformers` and PyTorch, which §9 forbids outright. Two are
supported, both one POST over `httpx` (§14.7) with the same request and response:
OpenRouter's `/rerank`, which takes any reranker it lists (`cohere/rerank-4-fast`,
`voyageai/rerank-3`, ...) on the OpenRouter key every other call already uses; and
Cohere's own API, the original `reranker: cohere`, on a separate Cohere key.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import NamedTuple

import httpx

from zotero_rag_core.config import COHERE_RERANKER, Config
from zotero_rag_core.embedding import MAX_RETRIES
from zotero_rag_core.errors import ZoteroRagError
from zotero_rag_core.store import StoredChunk

logger = logging.getLogger(__name__)

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
REQUEST_TIMEOUT_S = 30.0
#: Statuses worth another attempt: rate limits and transient server faults.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_DELAY_S = 8.0

#: The reranking network seam: (question, documents, top_n) -> [(index, score), ...],
#: best first, indices into `documents`.
RerankFn = Callable[[str, list[str], int], list[tuple[int, float]]]


class Ranking(list[tuple[int, float]]):
    """A reranker's (index, score) pairs, best first, and what the call cost.

    A list, so a reranker that only returns pairs still satisfies `RerankFn`.
    """

    cost: float | None = None


class RerankError(ZoteroRagError):
    """The reranker is misconfigured — a bad model name, or a rejected key."""


def build_reranker(config: Config, name: str) -> RerankFn:
    """A reranker for `name`: `cohere`, or a reranker id on OpenRouter.

    The credential is asserted here, at build time, so a misconfigured folder fails
    before a single question is embedded rather than part-way through a run.
    """
    if name == COHERE_RERANKER:
        config.global_.require_cohere()
        return _reranker(
            COHERE_RERANK_URL, config.global_.cohere_api_key, config.index.reranker_model, "Cohere"
        )
    config.global_.require_openrouter()
    return _reranker(
        f"{config.global_.endpoint.rstrip('/')}/rerank", config.global_.api_key, name, "OpenRouter"
    )


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """The provider's `Retry-After` when it gives one in seconds, else backoff."""
    try:
        return min(float(response.headers["retry-after"]), MAX_RETRY_DELAY_S)
    except (KeyError, ValueError):
        return min(0.5 * 2**attempt, MAX_RETRY_DELAY_S)


def _reranker(url: str, api_key: str, model: str, provider: str) -> RerankFn:
    def rerank(question: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        for attempt in range(MAX_RETRIES + 1):
            response = httpx.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "query": question,
                    "documents": documents,
                    "top_n": min(top_n, len(documents)),
                },
                timeout=REQUEST_TIMEOUT_S,
            )
            # Rate limits and server errors pass; several workers reranking at once
            # (`compare-embeddings`) or a free tier hit them routinely.
            if response.status_code not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
                break
            time.sleep(_retry_delay(response, attempt))

        # Any other 4xx is a configuration fault — an unknown model, a rejected key,
        # a malformed request — and it will fail identically for every remaining
        # question. Raising beats degrading silently 21 times in a row. A rate limit
        # that outlasted the retries is not one: it falls back like a 5xx.
        if 400 <= response.status_code < 500 and response.status_code != 429:
            raise RerankError(
                f"{provider} rejected the rerank request for model {model!r} "
                f"(HTTP {response.status_code}): {response.text[:200]}"
            )
        response.raise_for_status()

        payload = response.json()
        usage = payload.get("usage") or {}
        if usage:
            logger.debug("rerank usage with %s: %s", model, usage)
        results = payload.get("results") or []
        ranking = Ranking(
            (int(entry["index"]), float(entry.get("relevance_score", 0.0)))
            for entry in results
            if "index" in entry
        )
        # OpenRouter reports the charge; Cohere's own API reports billed units only.
        cost = usage.get("cost")
        ranking.cost = float(cost) if isinstance(cost, int | float) else None
        return ranking

    return rerank


def candidate_count(top_k: int, fanout: int) -> int:
    """How many candidates to retrieve so the reranker has something to choose from.

    A reranker handed exactly `top_k` candidates can only reorder them, which changes
    nothing about *which* passages the model sees — and the ordering of a small
    context matters far less than its contents. The gain comes from retrieving wider
    and letting the reranker discard.
    """
    return max(top_k * max(fanout, 1), top_k)


class RerankOutcome(NamedTuple):
    hits: list[StoredChunk]
    #: Whether reranking actually applied, rather than falling back.
    reranked: bool
    #: What the rerank call cost, when the provider reported it.
    cost: float | None = None


def apply_rerank(
    candidates: Sequence[StoredChunk],
    question: str,
    top_k: int,
    rerank: RerankFn | None,
) -> RerankOutcome:
    """Reorder `candidates` by relevance and cut to `top_k`.

    Returns the chunks, whether reranking actually applied, and its cost. A transient failure
    falls back to the retrieved order rather than losing the question (§10) — but it
    is reported as *not* reranked, because a run that silently half-reranked would
    make two `eval` reports incomparable while looking identical.

    A misconfiguration (`RerankError`) is deliberately *not* caught here: it would
    recur for every question, so the run should stop and say so.
    """
    ordered = list(candidates[:top_k])
    if rerank is None or not candidates:
        return RerankOutcome(ordered, False)

    try:
        ranked = rerank(question, [chunk.text for chunk in candidates], top_k)
    except RerankError:
        raise
    except Exception as exc:  # noqa: BLE001 - transport and provider errors vary
        logger.warning("reranking failed, keeping the retrieved order: %s", exc)
        return RerankOutcome(ordered, False)

    cost = ranked.cost if isinstance(ranked, Ranking) else None
    picked = [candidates[index] for index, _ in ranked if 0 <= index < len(candidates)]
    if not picked:
        # A provider that answers 200 with nothing usable must not empty the context.
        logger.warning("the reranker returned no usable result, keeping the retrieved order")
        return RerankOutcome(ordered, False, cost)

    logger.info("reranked %d candidate(s) down to %d", len(candidates), min(len(picked), top_k))
    return RerankOutcome(picked[:top_k], True, cost)
