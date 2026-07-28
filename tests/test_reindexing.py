"""M4: version-aware reindexing, repair, pruning, and the store fingerprint.

Real on-disk Chroma throughout; only the download and embedding seams are fakes.
Two embedding fakes are used deliberately — one 3-dimensional, one 4-dimensional —
because a model swap that keeps the vector width and one that changes it need
different remedies, and only the second can be caught by Chroma itself.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from zotero_rag.config import Config
from zotero_rag.fingerprint import StoreFingerprint
from zotero_rag.indexer import (
    find_orphans,
    index_plan,
    needs_reindex,
    prune_orphans,
    repair_embeddings,
)
from zotero_rag.planning import IndexPlan, PlannedAttachment
from zotero_rag.vectorstore import VectorStore

PdfFactory = Callable[[str, list[str]], Path]

MODEL_A = "openai/text-embedding-3-small"
MODEL_B = "acme/multilingual-embed"


def planned(key: str = "ATT1", title: str = "A Paper", version: int = 10) -> PlannedAttachment:
    return PlannedAttachment(
        attachment_key=key,
        item_key="ITEM1",
        title=title,
        authors="Doe, Jane",
        year="2021",
        doi="10.1/xyz",
        collection_name="Writing",
        zotero_url="https://www.zotero.org/u/items/ITEM1",
        version=version,
    )


def embed_3d(texts: list[str]) -> list[list[float]]:
    return [[float(len(text)), 1.0, 0.5] for text in texts]


def embed_4d(texts: list[str]) -> list[list[float]]:
    """A different model *and* a different vector width."""
    return [[float(len(text)), 1.0, 0.5, 0.25] for text in texts]


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma")


def run_index(
    store: VectorStore,
    attachments: list[PlannedAttachment],
    pdf: Path,
    model: str = MODEL_A,
    embed: Callable[[list[str]], list[list[float]]] = embed_3d,
) -> object:
    return index_plan(
        IndexPlan(collections=["Writing"], attachments=attachments),
        store=store,
        download=lambda key: pdf,
        embed=embed,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model=model,
    )


# --- what a chunk now records --------------------------------------------


def test_chunks_record_the_model_and_the_attachment_version(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    """These two fields are what make repair and version-aware reindexing possible."""
    run_index(store, [planned(version=7)], pdf_factory("doc.pdf", ["Only page"]))
    metadata = store.get_by_attachment("ATT1")[0].metadata
    assert metadata["embedding_model"] == MODEL_A
    assert metadata["attachment_version"] == 7


def test_the_store_records_and_returns_its_fingerprint(store: VectorStore) -> None:
    store.write_fingerprint(StoreFingerprint.intended(Config(), embedding_dim=3))
    recorded = store.read_fingerprint()
    assert recorded is not None
    assert recorded.embedding_model == MODEL_A
    assert recorded.embedding_dim == 3


def test_fingerprint_survives_reopening_the_store(tmp_path: Path) -> None:
    """It has to outlive the process, or it guards nothing."""
    directory = tmp_path / "chroma"
    VectorStore(directory).write_fingerprint(StoreFingerprint(embedding_model="x/y"))
    reopened = VectorStore(directory).read_fingerprint()
    assert reopened is not None and reopened.embedding_model == "x/y"


def test_dimension_is_derived_from_the_data_not_the_record(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    """The one fact that cannot be forged by editing a file."""
    assert store.derived_dimension() is None
    run_index(store, [planned()], pdf_factory("doc.pdf", ["Only page"]))
    assert store.derived_dimension() == 3


# --- version-aware reindexing --------------------------------------------


def test_unchanged_attachments_are_still_skipped(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    pdf = pdf_factory("doc.pdf", ["Some text"])
    run_index(store, [planned(version=5)], pdf)
    stats = run_index(store, [planned(version=5)], pdf)
    assert stats.already_indexed == 1  # type: ignore[attr-defined]
    assert stats.reindexed == 0  # type: ignore[attr-defined]


def test_a_newer_zotero_version_triggers_reindexing(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    """Acceptance for §11.1: a PDF replaced in Zotero is picked up."""
    run_index(store, [planned(version=5)], pdf_factory("old.pdf", ["Original text"]))
    replacement = pdf_factory("new.pdf", ["Replacement text"])

    stats = run_index(store, [planned(version=6)], replacement)

    assert stats.reindexed == 1  # type: ignore[attr-defined]
    assert stats.indexed == 0  # type: ignore[attr-defined]
    assert stats.already_indexed == 0  # type: ignore[attr-defined]
    stored = store.get_by_attachment("ATT1")
    assert "Replacement" in stored[0].text
    assert stored[0].metadata["attachment_version"] == 6


def test_reindexing_drops_the_pages_the_replacement_no_longer_has(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    """Without deleting first, a shorter replacement leaves orphaned tail pages."""
    run_index(store, [planned(version=1)], pdf_factory("long.pdf", ["One", "Two", "Three"]))
    assert store.count() == 3

    run_index(store, [planned(version=2)], pdf_factory("short.pdf", ["Only one now"]))

    assert store.count() == 1
    assert "Only one now" in store.get_by_attachment("ATT1")[0].text


def test_an_older_version_never_downgrades_the_store(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    run_index(store, [planned(version=9)], pdf_factory("doc.pdf", ["Newer"]))
    stats = run_index(store, [planned(version=4)], pdf_factory("other.pdf", ["Older"]))
    assert stats.already_indexed == 1  # type: ignore[attr-defined]
    assert "Newer" in store.get_by_attachment("ATT1")[0].text


def test_chunks_predating_version_tracking_are_left_alone(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    """Upgrading must not silently re-embed every existing store."""
    run_index(store, [planned(version=0)], pdf_factory("doc.pdf", ["Text"]))
    assert store.attachment_version("ATT1") == 0
    assert not needs_reindex(store, planned(version=0))
    # A real version arriving later does count as a change.
    assert needs_reindex(store, planned(version=3))


def test_needs_reindex_is_false_for_an_attachment_never_stored(store: VectorStore) -> None:
    assert not needs_reindex(store, planned(version=99))


# --- orphans -------------------------------------------------------------


def test_orphans_are_the_attachments_the_collection_no_longer_has(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    pdf = pdf_factory("doc.pdf", ["Text"])
    run_index(store, [planned("KEEP"), planned("GONE")], pdf)

    plan = IndexPlan(collections=["Writing"], attachments=[planned("KEEP")])

    assert find_orphans(store, plan) == {"GONE"}


def test_pruning_removes_only_the_orphaned_chunks(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    pdf = pdf_factory("doc.pdf", ["One", "Two"])
    run_index(store, [planned("KEEP"), planned("GONE")], pdf)
    assert store.count() == 4

    attachments, chunks = prune_orphans(store, IndexPlan(attachments=[planned("KEEP")]))

    assert (attachments, chunks) == (1, 2)
    assert store.count() == 2
    assert store.attachment_keys() == {"KEEP"}


def test_nothing_to_prune_is_a_no_op(store: VectorStore, pdf_factory: PdfFactory) -> None:
    run_index(store, [planned("KEEP")], pdf_factory("doc.pdf", ["Text"]))
    assert prune_orphans(store, IndexPlan(attachments=[planned("KEEP")])) == (0, 0)
    assert store.count() == 1


# --- repair --------------------------------------------------------------


def test_repair_re_embeds_only_the_stale_chunks(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    """The remedy for an edited `embedding_model`: no download, no re-chunking."""
    pdf = pdf_factory("doc.pdf", ["Page one", "Page two"])
    run_index(store, [planned("OLD")], pdf, model=MODEL_A)
    run_index(store, [planned("NEW")], pdf, model=MODEL_B)

    embedded: list[list[str]] = []

    def tracking_embed(texts: list[str]) -> list[list[float]]:
        embedded.append(texts)
        return embed_3d(texts)

    stats = repair_embeddings(store, tracking_embed, MODEL_B)

    assert stats.stale == 2  # the two chunks written with MODEL_A
    assert stats.repaired == 2
    assert stats.rebuilt is False
    assert store.stale_ids(MODEL_B) == []
    # The probe call plus one batch of exactly the stale texts.
    assert [len(batch) for batch in embedded] == [1, 2]


def test_repair_preserves_ids_and_text(store: VectorStore, pdf_factory: PdfFactory) -> None:
    """Chunk boundaries must survive: repair is not a re-chunk."""
    run_index(store, [planned()], pdf_factory("doc.pdf", ["Page one", "Page two"]), model=MODEL_A)
    before = {chunk.id: chunk.text for chunk in store.get_by_attachment("ATT1")}

    repair_embeddings(store, embed_3d, MODEL_B)

    after = {chunk.id: chunk.text for chunk in store.get_by_attachment("ATT1")}
    assert after == before
    assert store.count() == len(before)


def test_repair_rebuilds_the_store_when_the_width_changes(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    """A collection's dimension is fixed once written, so in-place upsert cannot work."""
    run_index(store, [planned()], pdf_factory("doc.pdf", ["Page one", "Page two"]), model=MODEL_A)

    stats = repair_embeddings(store, embed_4d, MODEL_B)

    assert stats.rebuilt is True
    assert stats.repaired == 2
    assert store.count() == 2
    assert store.derived_dimension() == 4
    assert store.stale_ids(MODEL_B) == []
    # And the store is queryable at the new width, which is the whole point.
    assert store.query([1.0, 1.0, 0.5, 0.25], 1)


def test_repair_treats_untagged_chunks_as_stale(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    """Pre-M4 chunks carry no model, and an unknown model cannot be assumed current.

    This pins verified chromadb behaviour: `$ne` matches documents missing the key.
    """
    run_index(store, [planned()], pdf_factory("doc.pdf", ["Text"]), model="")

    assert store.stale_ids(MODEL_A) != []
    stats = repair_embeddings(store, embed_3d, MODEL_A)
    assert stats.repaired == 1
    assert store.get_by_attachment("ATT1")[0].metadata["embedding_model"] == MODEL_A


def test_repair_is_a_no_op_when_everything_matches(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    run_index(store, [planned()], pdf_factory("doc.pdf", ["Text"]), model=MODEL_A)
    calls: list[list[str]] = []

    stats = repair_embeddings(store, lambda texts: calls.append(texts) or embed_3d(texts), MODEL_A)  # type: ignore[func-returns-value]

    assert stats == type(stats)()  # all zeros, nothing rebuilt
    assert calls == []  # not even the dimension probe


def test_repair_on_an_empty_store_does_nothing(store: VectorStore) -> None:
    assert repair_embeddings(store, embed_3d, MODEL_A).repaired == 0


# --- force ---------------------------------------------------------------


def test_reset_empties_the_store_but_keeps_it_usable(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    pdf = pdf_factory("doc.pdf", ["Text"])
    run_index(store, [planned()], pdf)

    store.reset()

    assert store.count() == 0
    # --force then re-embeds from cache, which must land in the fresh collection.
    stats = run_index(store, [planned()], pdf)
    assert stats.indexed == 1  # type: ignore[attr-defined]
    assert store.count() == 1


def test_reset_allows_a_different_vector_width(store: VectorStore, pdf_factory: PdfFactory) -> None:
    """This is why `--force` exists as well as `--repair`."""
    pdf = pdf_factory("doc.pdf", ["Text"])
    run_index(store, [planned()], pdf, model=MODEL_A, embed=embed_3d)
    store.reset()
    run_index(store, [planned()], pdf, model=MODEL_B, embed=embed_4d)
    assert store.derived_dimension() == 4
