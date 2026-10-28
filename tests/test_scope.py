"""Limiting a search to a folder subtree (PRD-CursesUI §4.3).

The store is a real Chroma instance throughout: the filter's correctness depends
on how Chroma matches inside list metadata, and the tempting wrong filters fail
silently rather than loudly, so only the real thing can catch them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from zotero_rag.errors import CollectionNotFoundError
from zotero_rag.querying import NO_CONTEXT_ANSWER, answer_question
from zotero_rag.retrieval import KeywordIndex, retrieve
from zotero_rag.snapshot import CollectionSnapshot
from zotero_rag.vectorstore import StoredChunk, VectorStore, in_scope, scope_where
from zotero_rag.zotero_api import CollectionNode

#: attachment → (collection keys, text, embedding). D predates `collection_keys`.
LIBRARY: dict[str, tuple[list[str] | None, str, list[float]]] = {
    "A": (["K1"], "Routing with ECMP across the fabric.", [1.0, 0.0, 0.0]),
    "B": (["K2"], "ECMP hashing and its load imbalance.", [0.9, 0.1, 0.0]),
    "C": (["K1", "K3"], "Congestion control for QUIC streams.", [0.0, 1.0, 0.0]),
    "D": (None, "An ECMP survey indexed before folders were recorded.", [0.95, 0.05, 0.0]),
}


def metadata(key: str, collection_keys: list[str] | None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "item_key": f"I{key}",
        "attachment_key": key,
        "title": f"Paper {key}",
        "page": 1,
        "chunk_index": 0,
    }
    if collection_keys is not None:
        meta["collection_keys"] = collection_keys
    return meta


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    store = VectorStore(tmp_path / "chroma")
    store.add(
        ids=[f"{key}:1:0" for key in LIBRARY],
        embeddings=[embedding for _, _, embedding in LIBRARY.values()],
        documents=[text for _, text, _ in LIBRARY.values()],
        metadatas=[metadata(key, keys) for key, (keys, _, _) in LIBRARY.items()],
    )
    return store


def attachments(chunks: list[StoredChunk]) -> set[str]:
    return {chunk.metadata["attachment_key"] for chunk in chunks}


# --- the filter itself -----------------------------------------------------


def test_one_key_uses_the_bare_contains_form() -> None:
    """Chroma refuses an `$or` of fewer than two clauses."""
    assert scope_where({"K1"}) == {"collection_keys": {"$contains": "K1"}}


def test_several_keys_become_a_sorted_or_of_contains() -> None:
    assert scope_where(["K2", "K1", "K2"]) == {
        "$or": [
            {"collection_keys": {"$contains": "K1"}},
            {"collection_keys": {"$contains": "K2"}},
        ]
    }


def test_an_empty_scope_is_refused_rather_than_sent() -> None:
    with pytest.raises(ValueError, match="empty"):
        scope_where(set())


def test_in_scope_applies_the_same_rule_in_python() -> None:
    assert in_scope({"collection_keys": ["K1", "K3"]}, {"K3"})
    assert not in_scope({"collection_keys": ["K1"]}, {"K2"})
    assert not in_scope({}, {"K1"})  # predates the field: outside every scope
    assert in_scope({}, None)  # ...but inside an unscoped search


def test_in_is_the_trap_this_module_avoids(store: VectorStore) -> None:
    """Pinned so that a Chroma upgrade that changes it is noticed.

    `$in` reads naturally as "any of these keys", but on a list value it matches
    nothing, with no error, which would make every folder look empty.
    """
    trap = cast(Any, {"collection_keys": {"$in": ["K1", "K2"]}})
    assert store._collection.get(where=trap)["ids"] == []


# --- vector search ---------------------------------------------------------


def test_an_unscoped_query_sees_the_whole_store(store: VectorStore) -> None:
    assert attachments(store.query([1.0, 0.0, 0.0], 10)) == {"A", "B", "C", "D"}


def test_a_scoped_query_returns_only_chunks_in_the_scope(store: VectorStore) -> None:
    assert attachments(store.query([1.0, 0.0, 0.0], 10, {"K1"})) == {"A", "C"}


def test_a_multi_key_scope_is_the_union(store: VectorStore) -> None:
    assert attachments(store.query([1.0, 0.0, 0.0], 10, {"K2", "K3"})) == {"B", "C"}


def test_the_scope_applies_before_the_cut(store: VectorStore) -> None:
    """B ranks below A and D overall; scoped to K2 it must still be returned."""
    assert attachments(store.query([1.0, 0.0, 0.0], 1, {"K2"})) == {"B"}


def test_a_scope_nobody_holds_returns_nothing(store: VectorStore) -> None:
    assert store.query([1.0, 0.0, 0.0], 10, {"NOPE"}) == []


def test_an_empty_scope_returns_nothing(store: VectorStore) -> None:
    assert store.query([1.0, 0.0, 0.0], 10, set()) == []


# --- keyword search and hybrid ------------------------------------------------


def test_bm25_keeps_only_in_scope_matches(store: VectorStore) -> None:
    index = KeywordIndex.from_store(store)
    assert attachments(index.search("ECMP", 10)) == {"A", "B", "D"}
    assert attachments(index.search("ECMP", 10, {"K2"})) == {"B"}


def test_bm25_fills_its_limit_from_the_scope(store: VectorStore) -> None:
    """Filtering after the cut would leave a narrow folder with nothing."""
    index = KeywordIndex.from_store(store)
    assert attachments(index.search("ECMP", 1, {"K2"})) == {"B"}


def test_bm25_with_an_empty_scope_returns_nothing(store: VectorStore) -> None:
    assert KeywordIndex.from_store(store).search("ECMP", 10, set()) == []


def test_hybrid_never_fuses_in_an_out_of_scope_chunk(store: VectorStore) -> None:
    index = KeywordIndex.from_store(store)
    hits = retrieve(store, [1.0, 0.0, 0.0], "ECMP", 10, "hybrid", index, scope={"K1"})
    assert attachments(hits) == {"A", "C"}


# --- the whole query --------------------------------------------------------


class Model:
    """Records the prompt it was given and cites passage 1."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, messages: list[BaseMessage]) -> AIMessage:
        self.prompts.append(str(messages[-1].content))
        return AIMessage(content="From the evidence [1].")


def ask(store: VectorStore, model: Model, **kwargs: Any) -> Any:
    return answer_question(
        "ECMP",
        store=store,
        embed_query=lambda question: [1.0, 0.0, 0.0],
        answer=model,
        top_k=10,
        **kwargs,
    )


def test_a_scoped_answer_draws_only_on_the_scope(store: VectorStore) -> None:
    model = Model()
    result = ask(store, model, scope={"K2"})
    assert {source.attachment_key for source in result.sources} == {"B"}
    assert "Paper A" not in model.prompts[0] and "Paper B" in model.prompts[0]


def test_a_scope_with_nothing_in_it_skips_the_model(store: VectorStore) -> None:
    model = Model()
    result = ask(store, model, scope={"NOPE"})
    assert result.answer == NO_CONTEXT_ANSWER
    assert model.prompts == []


def test_the_reranker_only_chooses_among_in_scope_candidates(store: VectorStore) -> None:
    seen: list[str] = []

    def rerank(question: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        seen.extend(documents)
        return [(index, 1.0) for index in range(min(top_n, len(documents)))]

    ask(store, Model(), scope={"K1"}, rerank=rerank)
    assert seen and all(text in (LIBRARY["A"][1], LIBRARY["C"][1]) for text in seen)


def test_no_scope_is_the_unscoped_query(store: VectorStore) -> None:
    """The default must leave `query` and `eval` exactly as they were."""
    unscoped = ask(store, Model())
    assert ask(store, Model(), scope=None).sources == unscoped.sources
    assert {source.attachment_key for source in unscoped.sources} == {"A", "B", "C", "D"}


# --- expanding a selected folder into a scope ---------------------------------


@pytest.fixture
def snapshot() -> CollectionSnapshot:
    return CollectionSnapshot(
        roots=[
            CollectionNode(
                key="NET",
                name="Networking",
                children=[
                    CollectionNode(key="K1", name="Routing"),
                    CollectionNode(
                        key="K2", name="Transport", children=[CollectionNode(key="K3", name="QUIC")]
                    ),
                ],
            )
        ]
    )


def test_a_folder_scope_includes_every_descendant(snapshot: CollectionSnapshot) -> None:
    assert snapshot.subtree_keys("K2") == {"K2", "K3"}
    assert snapshot.subtree_keys("NET") == {"NET", "K1", "K2", "K3"}


def test_a_leaf_folder_is_its_own_scope(snapshot: CollectionSnapshot) -> None:
    assert snapshot.subtree_keys("K3") == {"K3"}


def test_an_unknown_folder_is_an_error(snapshot: CollectionSnapshot) -> None:
    with pytest.raises(CollectionNotFoundError):
        snapshot.subtree_keys("GONE")


def test_selecting_a_parent_folder_finds_items_filed_only_below_it(
    store: VectorStore, snapshot: CollectionSnapshot
) -> None:
    """C is filed in QUIC (K3), not in Transport (K2) itself, yet Transport finds it."""
    hits = store.query([0.0, 1.0, 0.0], 10, snapshot.subtree_keys("K2"))
    assert attachments(hits) == {"B", "C"}
