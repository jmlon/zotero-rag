"""The `ChunkStore` contract, run against every implementation.

The indexer, retrieval, querying and evaluation are written against the protocol,
so a store that behaves differently would change answers without changing any of
their code. Each test here runs on the Chroma store an index folder keeps and on
the PostgreSQL store a hosted deployment uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zotero_rag_core.fingerprint import StoreFingerprint
from zotero_rag_core.indexer import repair_embeddings
from zotero_rag_core.planning import PlannedAttachment
from zotero_rag_core.retrieval import BM25Searcher, retrieve
from zotero_rag_core.store import ChunkStore, Filters, VectorStoreError
from zotero_rag_core.stores.chroma import ChromaStore


@pytest.fixture(params=["chroma", "pg"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ChunkStore:
    if request.param == "chroma":
        return ChromaStore(tmp_path / "chroma")
    pg: ChunkStore = request.getfixturevalue("pg_store")
    return pg


def attachment(key: str, item: str = "", **fields: Any) -> PlannedAttachment:
    base: dict[str, Any] = {
        "attachment_key": key,
        "item_key": item or f"I{key}",
        "item_type": "journalArticle",
        "title": f"Paper {key}",
        "authors": "Doe, Jane",
        "year": "2021",
        "doi": "10.1/x",
        "collection_name": "Writing",
        "collection_keys": ["K1"],
        "tags": ["RAG"],
        "zotero_url": f"https://www.zotero.org/u/items/{key}",
        "version": 7,
    }
    return PlannedAttachment(**(base | fields))


def put(
    store: ChunkStore,
    planned: PlannedAttachment,
    pages: dict[int, list[str]],
    vectors: dict[int, list[float]] | None = None,
    model: str = "m1",
) -> list[str]:
    """Store one attachment's chunks as the indexer would; returns their ids."""
    ids, texts, embeddings, metadatas = [], [], [], []
    for page, texts_on_page in pages.items():
        for index, text in enumerate(texts_on_page):
            ids.append(f"{planned.attachment_key}:{page}:{index}")
            texts.append(text)
            embeddings.append((vectors or {}).get(page, [1.0, float(page), 0.0]))
            metadatas.append(planned.chunk_metadata(page, index, model))
    store.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return ids


# --- writing and reading back --------------------------------------------------------


def test_a_chunk_reads_back_as_it_was_written(store: ChunkStore) -> None:
    planned = attachment("A", tags=["LLM", "RAG"])
    put(store, planned, {3: ["Some passage."]})
    (chunk,) = store.get_by_ids(["A:3:0"])
    assert chunk.text == "Some passage."
    assert chunk.metadata == planned.chunk_metadata(3, 0, "m1")


def test_metadata_beyond_the_known_fields_survives(store: ChunkStore) -> None:
    metadata = attachment("A").chunk_metadata(1, 0, "m1") | {"custom": "kept", "n": 3}
    store.add(ids=["A:1:0"], embeddings=[[1.0, 0.0, 0.0]], documents=["t"], metadatas=[metadata])
    assert store.get_by_ids(["A:1:0"])[0].metadata == metadata


def test_counts_keys_and_versions(store: ChunkStore) -> None:
    put(store, attachment("A"), {1: ["a", "b"], 2: ["c"]})
    put(store, attachment("B", version=9), {1: ["d"]})
    assert store.count() == 4
    assert store.attachment_keys() == {"A", "B"}
    assert store.has_attachment("A") and not store.has_attachment("Z")
    assert store.attachment_version("B") == 9
    assert store.attachment_version("Z") is None


def test_adding_an_existing_id_keeps_the_stored_chunk(store: ChunkStore) -> None:
    put(store, attachment("A"), {1: ["first"]})
    put(store, attachment("A"), {1: ["second"]})
    assert [chunk.text for chunk in store.get_by_ids(["A:1:0"])] == ["first"]


def test_upsert_replaces_text_vector_and_metadata(store: ChunkStore) -> None:
    put(store, attachment("A"), {1: ["first"]})
    metadata = attachment("A").chunk_metadata(1, 0, "m2")
    store.upsert(
        ids=["A:1:0"], embeddings=[[0.0, 1.0, 0.0]], documents=["second"], metadatas=[metadata]
    )
    (chunk,) = store.get_by_ids(["A:1:0"])
    assert chunk.text == "second" and chunk.metadata["embedding_model"] == "m2"
    assert store.query([0.0, 1.0, 0.0], 1)[0].id == "A:1:0"


def test_an_attachment_reads_in_page_order(store: ChunkStore) -> None:
    """Page 10 after page 2, which sorting ids as strings would get wrong."""
    put(store, attachment("A"), {10: ["ten"], 2: ["two-a", "two-b"]})
    assert [chunk.text for chunk in store.get_by_attachment("A")] == ["two-a", "two-b", "ten"]


def test_deleting_an_attachment_removes_only_its_chunks(store: ChunkStore) -> None:
    put(store, attachment("A"), {1: ["a"], 2: ["b"]})
    put(store, attachment("B"), {1: ["c"]})
    assert store.delete_by_attachment("A") == 2
    assert store.attachment_keys() == {"B"}
    assert store.delete_by_attachment("A") == 0


def test_deleting_ids(store: ChunkStore) -> None:
    put(store, attachment("A"), {1: ["a"], 2: ["b"]})
    store.delete_ids(["A:1:0"])
    store.delete_ids([])
    assert [chunk.id for chunk in store.all_chunks()] == ["A:2:0"]


def test_moving_an_attachment_between_folders_needs_no_re_embedding(store: ChunkStore) -> None:
    put(store, attachment("A"), {1: ["a"], 2: ["b"]})
    assert store.set_collection_keys("A", ["K9", "K2"]) == 2
    assert store.collection_keys_by_attachment() == {"A": ["K2", "K9"]}
    assert store.set_collection_keys("Z", ["K1"]) == 0
    with pytest.raises(ValueError):
        store.set_collection_keys("A", [])


def test_stale_ids_are_those_of_another_model(store: ChunkStore) -> None:
    put(store, attachment("A"), {1: ["a"]}, model="old")
    put(store, attachment("B"), {1: ["b"]}, model="new")
    assert store.stale_ids("new") == ["A:1:0"]


# --- searching -------------------------------------------------------------------------


def test_the_nearest_chunks_come_first_and_top_k_cuts(store: ChunkStore) -> None:
    put(
        store,
        attachment("A"),
        {1: ["near"], 2: ["middle"], 3: ["far"]},
        vectors={1: [1.0, 0.0, 0.0], 2: [0.7, 0.7, 0.0], 3: [0.0, 0.0, 1.0]},
    )
    hits = store.query([1.0, 0.05, 0.0], 2)
    assert [hit.text for hit in hits] == ["near", "middle"]
    assert hits[0].distance is not None and hits[0].distance <= (hits[1].distance or 0)


def test_filters_apply_before_the_cut(store: ChunkStore) -> None:
    put(store, attachment("A"), {1: ["near"]}, vectors={1: [1.0, 0.0, 0.0]})
    put(store, attachment("B", collection_keys=["K2"]), {1: ["far"]}, vectors={1: [0.0, 1.0, 0.0]})
    hits = store.query([1.0, 0.0, 0.0], 1, Filters(collection_keys=frozenset({"K2"})))
    assert [hit.text for hit in hits] == ["far"]


def test_hybrid_retrieval_runs_on_the_store(store: ChunkStore) -> None:
    put(
        store,
        attachment("A"),
        {1: ["Load balancing in general."], 2: ["ECMP spreads flows."]},
        vectors={1: [1.0, 0.0, 0.0], 2: [0.0, 0.0, 1.0]},
    )
    hits = retrieve(store, [1.0, 0.0, 0.0], "ECMP", 2, "hybrid", BM25Searcher.from_store(store))
    assert {hit.text for hit in hits} == {"Load balancing in general.", "ECMP spreads flows."}


# --- how the store was built -------------------------------------------------------------


def test_the_fingerprint_is_recorded(store: ChunkStore) -> None:
    assert store.read_fingerprint() is None
    fingerprint = StoreFingerprint(embedding_model="m1", embedding_dim=3, chunk_size=1000)
    store.write_fingerprint(fingerprint)
    assert store.read_fingerprint() == fingerprint


def test_the_dimension_is_read_from_the_data(store: ChunkStore) -> None:
    assert store.derived_dimension() is None
    put(store, attachment("A"), {1: ["a"]})
    assert store.derived_dimension() == 3


def test_reset_empties_the_store_for_another_width(store: ChunkStore) -> None:
    put(store, attachment("A"), {1: ["a"]})
    store.write_fingerprint(StoreFingerprint(embedding_model="m1"))
    store.reset()
    assert store.count() == 0 and store.read_fingerprint() is None
    put(store, attachment("A"), {1: ["a"]}, vectors={1: [1.0, 0.0, 0.0, 0.0, 0.0]})
    assert store.derived_dimension() == 5


def test_repair_rebuilds_a_store_for_a_model_of_another_width(store: ChunkStore) -> None:
    put(store, attachment("A"), {1: ["a"], 2: ["b"]}, model="old")

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    stats = repair_embeddings(store, embed, "new")
    assert stats.rebuilt and stats.repaired == 2
    assert store.derived_dimension() == 4
    assert store.stale_ids("new") == []
    assert [chunk.text for chunk in store.get_by_attachment("A")] == ["a", "b"]


def test_vectors_of_two_widths_are_refused(store: ChunkStore) -> None:
    """Mixing two models' vectors must fail loudly, not search nonsense."""
    put(store, attachment("A"), {1: ["a"]})
    with pytest.raises((VectorStoreError, ValueError)):
        put(store, attachment("B"), {1: ["b"]}, vectors={1: [1.0, 0.0]})
