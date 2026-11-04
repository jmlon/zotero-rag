"""Retrieval diversity: a soft per-document cap on the hits (M7, §11.4).

§11.4.1 recorded one question whose thirty hits all came from a single document while
the paper it was about never appeared. The cap is checked here on its own, then
through the query pipeline against a real Chroma store, with and without a reranker.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from zotero_rag_core.config import IndexConfig
from zotero_rag_core.evaluation import run_settings
from zotero_rag_core.querying import QueryResult, answer_question
from zotero_rag_core.retrieval import candidate_depth, cap_per_document, document_key
from zotero_rag_core.store import StoredChunk
from zotero_rag_core.stores.chroma import ChromaStore


def chunk(identifier: str, item: str | None = None, attachment: str | None = None) -> StoredChunk:
    metadata: dict[str, str] = {}
    if item is not None:
        metadata["item_key"] = item
    if attachment is not None:
        metadata["attachment_key"] = attachment
    return StoredChunk(id=identifier, text=identifier, metadata=metadata)


def ids(chunks: list[StoredChunk]) -> list[str]:
    return [c.id for c in chunks]


# --- which document a chunk belongs to ------------------------------------


def test_a_chunk_belongs_to_its_zotero_item() -> None:
    assert document_key(chunk("x", item="ITEM", attachment="PDF")) == "ITEM"


def test_without_an_item_key_the_attachment_stands_in() -> None:
    assert document_key(chunk("x", attachment="PDF")) == "PDF"


def test_a_chunk_with_no_metadata_is_its_own_document() -> None:
    assert document_key(chunk("A:1:0")) == "A:1:0"
    assert document_key(chunk("A:1:1")) != document_key(chunk("A:1:0"))


# --- the cap on its own ---------------------------------------------------


def test_a_cap_of_zero_is_a_plain_cut() -> None:
    ranked = [chunk("a1", "A"), chunk("a2", "A"), chunk("b1", "B")]
    assert ids(cap_per_document(ranked, 2, 0)) == ["a1", "a2"]


def test_the_cap_promotes_other_documents_in_rank_order() -> None:
    ranked = [chunk("a1", "A"), chunk("a2", "A"), chunk("a3", "A"), chunk("b1", "B")]
    ranked += [chunk("c1", "C")]
    assert ids(cap_per_document(ranked, 3, 1)) == ["a1", "b1", "c1"]


def test_a_cap_above_one_keeps_that_many_per_document() -> None:
    ranked = [chunk("a1", "A"), chunk("a2", "A"), chunk("a3", "A"), chunk("b1", "B")]
    assert ids(cap_per_document(ranked, 3, 2)) == ["a1", "a2", "b1"]


def test_attachments_of_one_item_share_its_quota() -> None:
    """A PDF and a snapshot of the same paper are one document to the reader."""
    ranked = [
        chunk("pdf", item="PAPER", attachment="PDF"),
        chunk("html", item="PAPER", attachment="SNAPSHOT"),
        chunk("other", item="OTHER", attachment="X"),
    ]
    assert ids(cap_per_document(ranked, 2, 1)) == ["pdf", "other"]


def test_the_cap_is_soft_when_one_document_is_all_there_is() -> None:
    """A folder holding one paper must still fill top_k, not starve the model."""
    ranked = [chunk("a1", "A"), chunk("a2", "A"), chunk("a3", "A")]
    assert ids(cap_per_document(ranked, 3, 1)) == ["a1", "a2", "a3"]


def test_refilled_slots_go_to_the_best_skipped_chunks_in_rank_order() -> None:
    ranked = [chunk("a1", "A"), chunk("a2", "A"), chunk("a3", "A"), chunk("b1", "B")]
    # One A and one B fit the cap; the third slot goes to a2, the best skipped one.
    assert ids(cap_per_document(ranked, 3, 1)) == ["a1", "a2", "b1"]


def test_the_cap_never_returns_more_than_the_limit_or_the_candidates() -> None:
    ranked = [chunk("a1", "A"), chunk("b1", "B")]
    assert ids(cap_per_document(ranked, 5, 1)) == ["a1", "b1"]
    assert cap_per_document([], 5, 1) == []


# --- through the query pipeline ------------------------------------------


@pytest.fixture
def crowded_store(tmp_path: Path) -> ChromaStore:
    """Paper A is nearest the question everywhere; paper B is the one that answers."""
    store = ChromaStore(tmp_path / "chroma")
    a_ids = [f"A:1:{index}" for index in range(4)]
    store.add(
        ids=[*a_ids, "B:1:0"],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
            [0.97, 0.03, 0.0],
            [0.6, 0.4, 0.0],
        ],
        documents=["a zero", "a one", "a two", "a three", "the answer in B"],
        metadatas=[
            *(
                {"item_key": "A", "attachment_key": "A", "page": 1, "chunk_index": index}
                for index in range(4)
            ),
            {"item_key": "B", "attachment_key": "B", "page": 1, "chunk_index": 0},
        ],
    )
    return store


def reply(_: object) -> AIMessage:
    return AIMessage(
        content="Answered [1].",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )


def ask(store: ChromaStore, **kwargs: object) -> QueryResult:
    return answer_question(
        "where is the answer?",
        store=store,
        embed_query=lambda _: [1.0, 0.0, 0.0],
        answer=reply,
        **kwargs,  # type: ignore[arg-type]
    )


def texts(result: QueryResult) -> list[str]:
    return [excerpt for source in result.sources for excerpt in source.excerpts]


def test_without_a_cap_one_paper_crowds_out_the_answer(crowded_store: ChromaStore) -> None:
    """The failure §11.4.1 recorded, reproduced, and the pre-M7 depth unchanged."""
    result = ask(crowded_store, top_k=2)
    assert "the answer in B" not in texts(result)
    assert result.candidates_considered == 2


def test_a_cap_lets_the_answering_paper_in(crowded_store: ChromaStore) -> None:
    result = ask(crowded_store, top_k=2, max_per_document=1)

    assert result.chunks_retrieved == 2
    assert texts(result) == ["a zero", "the answer in B"]
    # It retrieved deeper than top_k, which is what gave it B to promote.
    assert result.candidates_considered == min(candidate_depth(2), 5)


def test_the_cap_applies_after_reranking(crowded_store: ChromaStore) -> None:
    """The reranker picks *which* passage of A survives; the cap only limits how many."""
    seen: list[int] = []

    def rerank(question: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        seen.append(top_n)
        order = [3, 2, 1, 0, 4]  # a three, a two, a one, a zero, B
        return [(index, 1.0 - rank / 10) for rank, index in enumerate(order)]

    result = ask(crowded_store, top_k=2, max_per_document=1, rerank=rerank, rerank_fanout=4)

    assert result.reranked is True
    assert texts(result) == ["a three", "the answer in B"]
    # Every candidate came back ranked, not just top_k, so the cap had B to promote.
    assert seen == [5]


def test_window_neighbours_do_not_count_against_the_cap(crowded_store: ChromaStore) -> None:
    result = ask(crowded_store, top_k=2, max_per_document=1, window=1)

    assert result.chunks_retrieved == 2
    # a zero's neighbour a one arrives as context even though A's quota is spent.
    assert {"a zero", "a one", "the answer in B"} <= set(texts(result))


# --- configuration and reporting ------------------------------------------


def test_the_cap_is_off_by_default() -> None:
    assert IndexConfig().max_per_document == 0


def test_a_negative_cap_is_rejected() -> None:
    with pytest.raises(ValidationError):
        IndexConfig(max_per_document=-1)


def test_eval_settings_record_the_cap(crowded_store: ChromaStore, tmp_path: Path) -> None:
    from zotero_rag.config import Config

    settings = run_settings(
        Config(), crowded_store, 2, tmp_path / "questions.yaml", max_per_document=1
    )
    assert settings.max_per_document == 1
