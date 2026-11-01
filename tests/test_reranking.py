"""Reranking retrieved candidates (§11.4.3).

The Cohere call is a local callable, as every network seam in this codebase is. The
HTTP layer itself is exercised against a stub transport, so the request body and the
4xx-versus-5xx policy are checked without reaching the provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from zotero_rag.config import Config
from zotero_rag_core import reranking
from zotero_rag_core.embedding import MAX_RETRIES
from zotero_rag_core.errors import MissingCredentialError
from zotero_rag_core.querying import answer_question
from zotero_rag_core.reranking import (
    COHERE_RERANK_URL,
    RerankError,
    apply_rerank,
    build_reranker,
    candidate_count,
)
from zotero_rag_core.vectorstore import StoredChunk, VectorStore


def chunk(identifier: str, text: str) -> StoredChunk:
    return StoredChunk(
        id=identifier,
        text=text,
        metadata={"attachment_key": identifier.split(":")[0], "page": 1, "chunk_index": 0},
    )


CANDIDATES = [
    chunk("A:1:0", "about fog nodes"),
    chunk("B:1:0", "about mercury"),
    chunk("C:1:0", "about edge computing"),
    chunk("D:1:0", "about routing"),
]


def reranker(order: list[int]):  # type: ignore[no-untyped-def]
    """A reranker returning `order` as its ranking, best first."""

    def rerank(question: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        return [(index, 1.0 - position / 10) for position, index in enumerate(order)]

    return rerank


# --- fanout --------------------------------------------------------------


def test_fanout_widens_the_candidate_pool() -> None:
    """A reranker handed exactly top_k can only reorder, not discard."""
    assert candidate_count(10, 4) == 40
    assert candidate_count(5, 1) == 5
    # A nonsense fanout must not shrink the pool below what is being asked for.
    assert candidate_count(10, 0) == 10


# --- applying it ---------------------------------------------------------


def test_reranking_reorders_and_cuts() -> None:
    picked, reranked, _ = apply_rerank(CANDIDATES, "fog?", top_k=2, rerank=reranker([2, 0, 3, 1]))

    assert reranked is True
    assert [c.id for c in picked] == ["C:1:0", "A:1:0"]


def test_reranking_can_discard_a_candidate_retrieval_ranked_first() -> None:
    """The whole point: relevance to the question, not similarity of the embedding."""
    picked, _, _ = apply_rerank(CANDIDATES, "fog?", top_k=1, rerank=reranker([2]))
    assert [c.id for c in picked] == ["C:1:0"]


def test_no_reranker_keeps_the_retrieved_order() -> None:
    picked, reranked, _ = apply_rerank(CANDIDATES, "fog?", top_k=2, rerank=None)
    assert [c.id for c in picked] == ["A:1:0", "B:1:0"]
    assert reranked is False


def test_no_candidates_is_not_an_error() -> None:
    assert apply_rerank([], "fog?", top_k=5, rerank=reranker([0])) == ([], False, None)


def test_a_transient_failure_falls_back_to_the_retrieved_order() -> None:
    """Losing the question would be worse than answering from the unreranked order."""

    def failing(question: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        raise httpx.ReadTimeout("too slow")

    picked, reranked, _ = apply_rerank(CANDIDATES, "fog?", top_k=2, rerank=failing)

    assert [c.id for c in picked] == ["A:1:0", "B:1:0"]
    # Reported as not reranked, so a half-reranked run cannot pass for a reranked one.
    assert reranked is False


def test_a_misconfiguration_is_raised_not_swallowed() -> None:
    """A bad model name fails identically for every question; stop and say so."""

    def misconfigured(question: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        raise RerankError("Cohere rejected the rerank request")

    with pytest.raises(RerankError):
        apply_rerank(CANDIDATES, "fog?", top_k=2, rerank=misconfigured)


def test_an_empty_ranking_does_not_empty_the_context() -> None:
    """A provider answering 200 with nothing usable must not lose every passage."""
    picked, reranked, _ = apply_rerank(CANDIDATES, "fog?", top_k=2, rerank=reranker([]))
    assert [c.id for c in picked] == ["A:1:0", "B:1:0"]
    assert reranked is False


def test_out_of_range_indices_are_ignored() -> None:
    picked, reranked, _ = apply_rerank(CANDIDATES, "fog?", top_k=3, rerank=reranker([99, 1, -1, 0]))
    assert [c.id for c in picked] == ["B:1:0", "A:1:0"]
    assert reranked is True


# --- the Cohere client ---------------------------------------------------


def cohere_config(key: str = "test-key") -> Config:
    config = Config()
    config.global_.cohere_api_key = key
    return config


def stub_transport(handler):  # type: ignore[no-untyped-def]
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The backoff between retries, recorded instead of slept."""
    slept: list[float] = []
    monkeypatch.setattr(reranking.time, "sleep", slept.append)
    return slept


def test_building_a_reranker_without_a_key_fails_early() -> None:
    """Before a single question is embedded, not part-way through a run."""
    with pytest.raises(MissingCredentialError, match="COHERE_API_KEY"):
        build_reranker(cohere_config(key=""), "cohere")


def test_the_request_carries_the_model_query_and_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.9}]})

    transport = stub_transport(handler)
    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: httpx.Client(transport=transport).post(url, **kw)
    )

    rerank = build_reranker(cohere_config(), "cohere")
    result = rerank("what is fog?", ["doc one", "doc two"], 2)

    assert result == [(1, 0.9)]
    assert seen["url"] == COHERE_RERANK_URL
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "rerank-v4.0-fast"
    assert body["query"] == "what is fog?"
    assert body["documents"] == ["doc one", "doc two"]
    # top_n is clamped: asking for more than were sent is an error at the provider.
    assert body["top_n"] == 2


def test_top_n_is_clamped_to_the_documents_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    transport = stub_transport(handler)
    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: httpx.Client(transport=transport).post(url, **kw)
    )

    build_reranker(cohere_config(), "cohere")("q", ["only one"], 10)

    body = seen["body"]
    assert isinstance(body, dict)
    assert body["top_n"] == 1


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_a_client_error_is_a_configuration_fault(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It would fail the same way for every question, so it must not degrade quietly."""
    transport = stub_transport(lambda request: httpx.Response(status, text="model not found"))
    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: httpx.Client(transport=transport).post(url, **kw)
    )

    with pytest.raises(RerankError, match="rerank-v4.0-fast"):
        build_reranker(cohere_config(), "cohere")("q", ["a", "b"], 2)


def test_a_server_error_stays_transient(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """A 503 may well succeed next time, so it is not a RerankError."""
    transport = stub_transport(lambda request: httpx.Response(503, text="try later"))
    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: httpx.Client(transport=transport).post(url, **kw)
    )

    rerank = build_reranker(cohere_config(), "cohere")
    with pytest.raises(httpx.HTTPStatusError):
        rerank("q", ["a", "b"], 2)
    assert len(sleeps) == MAX_RETRIES

    # And through apply_rerank it degrades rather than failing the question.
    picked, reranked, _ = apply_rerank(CANDIDATES, "q", top_k=2, rerank=rerank)
    assert len(picked) == 2 and reranked is False


def test_a_malformed_result_entry_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = stub_transport(
        lambda request: httpx.Response(
            200, json={"results": [{"relevance_score": 0.5}, {"index": 0}]}
        )
    )
    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: httpx.Client(transport=transport).post(url, **kw)
    )

    assert build_reranker(cohere_config(), "cohere")("q", ["a"], 1) == [(0, 0.0)]


# --- the OpenRouter client ------------------------------------------------


def openrouter_config(key: str = "router-key") -> Config:
    config = Config()
    config.global_.api_key = key
    return config


def responding(monkeypatch: pytest.MonkeyPatch, *responses: httpx.Response) -> list[dict]:  # type: ignore[type-arg]
    """Serve `responses` in turn (the last one repeats); returns each request seen."""
    seen: list[dict] = []  # type: ignore[type-arg]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("Authorization"),
                "body": json.loads(request.content),
            }
        )
        return responses[min(len(seen), len(responses)) - 1]

    transport = stub_transport(handler)
    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: httpx.Client(transport=transport).post(url, **kw)
    )
    return seen


def test_an_openrouter_reranker_calls_the_openrouter_endpoint_with_its_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = responding(
        monkeypatch,
        httpx.Response(
            200,
            json={
                "results": [{"index": 1, "relevance_score": 0.2}],
                "usage": {"total_tokens": 56, "cost": 0},
            },
        ),
    )

    result = build_reranker(openrouter_config(), "voyageai/rerank-3")("fog?", ["a", "b"], 1)

    assert result == [(1, 0.2)]
    assert seen[0]["url"] == "https://openrouter.ai/api/v1/rerank"
    # The OpenRouter key, not a Cohere one: no second provider to sign up for.
    assert seen[0]["auth"] == "Bearer router-key"
    assert seen[0]["body"]["model"] == "voyageai/rerank-3"
    assert seen[0]["body"]["top_n"] == 1


def test_an_openrouter_reranker_needs_only_the_openrouter_key() -> None:
    with pytest.raises(MissingCredentialError, match="OPENROUTER_API_KEY"):
        build_reranker(openrouter_config(key=""), "cohere/rerank-4-fast")
    # No Cohere key is set, and none is asked for.
    build_reranker(openrouter_config(), "cohere/rerank-4-fast")


def test_a_rate_limit_is_waited_out(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    seen = responding(
        monkeypatch,
        httpx.Response(429, headers={"Retry-After": "3"}, text="slow down"),
        httpx.Response(429, text="slow down"),
        httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]}),
    )

    result = build_reranker(openrouter_config(), "cohere/rerank-4-fast")("q", ["a"], 1)

    assert result == [(0, 0.9)]
    assert len(seen) == 3
    # The provider's own wait first, then exponential backoff.
    assert sleeps == [3.0, 1.0]


def test_a_rate_limit_that_outlasts_the_retries_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike a bad model name, a rate limit will pass: degrade, don't stop the run."""
    seen = responding(monkeypatch, httpx.Response(429, text="slow down"))
    rerank = build_reranker(openrouter_config(), "cohere/rerank-4-fast")

    with pytest.raises(httpx.HTTPStatusError):
        rerank("q", ["a", "b"], 2)
    assert len(seen) == MAX_RETRIES + 1

    picked, reranked, _ = apply_rerank(CANDIDATES, "q", top_k=2, rerank=rerank)
    assert len(picked) == 2 and reranked is False


def test_an_unknown_openrouter_model_is_a_configuration_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = responding(monkeypatch, httpx.Response(404, text="Model does not exist"))

    with pytest.raises(RerankError, match="OpenRouter rejected .*nobody/reranker"):
        build_reranker(openrouter_config(), "nobody/reranker")("q", ["a"], 1)
    assert len(seen) == 1


def test_reranker_names_are_validated_in_the_config() -> None:
    assert Config.model_validate({"index": {"reranker": "cohere/rerank-4-fast"}}).reranker == (
        "cohere/rerank-4-fast"
    )
    with pytest.raises(ValueError, match="unknown reranker 'magic'"):
        Config.model_validate({"index": {"reranker": "magic"}})


def test_the_reranker_model_follows_the_reranker() -> None:
    config = Config()
    assert config.reranker_model_for("cohere") == config.index.reranker_model
    assert config.reranker_model_for("voyageai/rerank-3") == "voyageai/rerank-3"


# --- through the query pipeline ------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    """Four chunks; the vector order deliberately disagrees with relevance."""
    store = VectorStore(tmp_path / "chroma")
    store.add(
        ids=["A:1:0", "A:1:1", "A:1:2", "A:1:3"],
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0]],
        documents=["nearest but useless", "second", "third", "the actual answer"],
        metadatas=[{"attachment_key": "A", "page": 1, "chunk_index": index} for index in range(4)],
    )
    return store


def reply():  # type: ignore[no-untyped-def]
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="Because [1].",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )


def test_reranking_retrieves_wide_then_cuts(store: VectorStore) -> None:
    """`top_k` still bounds what reaches the model; the pool behind it grew."""
    result = answer_question(
        "which one answers it?",
        store=store,
        embed_query=lambda _: [1.0, 0.0, 0.0],
        answer=lambda _: reply(),
        top_k=1,
        rerank=reranker([3]),
        rerank_fanout=4,
    )

    assert result.candidates_considered == 4
    assert result.chunks_retrieved == 1
    assert result.reranked is True
    # The passage the vectors ranked last is the one that got through.
    assert result.sources[0].excerpts == ["the actual answer"]


def test_without_a_reranker_the_pool_is_just_top_k(store: VectorStore) -> None:
    result = answer_question(
        "anything",
        store=store,
        embed_query=lambda _: [1.0, 0.0, 0.0],
        answer=lambda _: reply(),
        top_k=2,
    )
    assert result.candidates_considered == 2
    assert result.reranked is False


def test_reranking_runs_before_windowing(store: VectorStore) -> None:
    """Neighbours are context for the chosen hit, so they must not be reranked away."""
    result = answer_question(
        "which one?",
        store=store,
        embed_query=lambda _: [1.0, 0.0, 0.0],
        answer=lambda _: reply(),
        top_k=1,
        window=1,
        rerank=reranker([3]),
        rerank_fanout=4,
    )

    assert result.chunks_retrieved == 1
    # The chosen chunk plus its neighbour, which the reranker never saw as a rival.
    assert result.chunks_in_context == 2
    texts = [excerpt for source in result.sources for excerpt in source.excerpts]
    assert "the actual answer" in texts
    assert "third" in texts


def test_the_reported_cost_travels_with_the_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ],
                "usage": {"total_tokens": 10, "cost": 2e-07},
            },
        )

    transport = stub_transport(handler)
    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: httpx.Client(transport=transport).post(url, **kw)
    )
    rerank = build_reranker(cohere_config(), "cohere")

    outcome = apply_rerank(CANDIDATES[:2], "fog?", top_k=1, rerank=rerank)

    assert outcome.reranked is True
    assert outcome.cost == 2e-07
    assert outcome.hits == [CANDIDATES[1]]


def test_a_reranker_that_reports_no_cost_is_unpriced() -> None:
    outcome = apply_rerank(CANDIDATES, "fog?", top_k=2, rerank=reranker([2, 0]))
    assert outcome.reranked is True
    assert outcome.cost is None
