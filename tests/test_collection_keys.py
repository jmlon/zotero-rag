"""`collection_keys` in the store: written at index time, kept current, backfilled.

Subtree filtering in the terminal UI depends on it (PRD-CursesUI §4.3). These tests
use a real Chroma store, because the guarantees being relied on (lists accepted,
`update` merging rather than replacing) are Chroma's, not ours.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from zotero_rag_core.indexer import index_plan, refresh_collection_keys
from zotero_rag_core.planning import IndexPlan, PlannedAttachment
from zotero_rag_core.stores.chroma import ChromaStore

PdfFactory = Callable[[str, list[str]], Path]


def planned(key: str = "ATT1", collection_keys: list[str] | None = None) -> PlannedAttachment:
    return PlannedAttachment(
        attachment_key=key,
        item_key=f"ITEM-{key}",
        title="A Paper",
        authors="Doe, Jane",
        year="2021",
        collection_name="Writing",
        collection_keys=collection_keys if collection_keys is not None else ["K1"],
        version=10,
    )


class CountingEmbedder:
    def __init__(self) -> None:
        self.texts = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.texts += len(texts)
        return [[float(len(text)), 1.0, 0.5] for text in texts]


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(tmp_path / "chroma")


@pytest.fixture
def pdf(pdf_factory: PdfFactory) -> Path:
    return pdf_factory("doc.pdf", ["Page one text", "Page two text"])


def index(store: ChromaStore, attachments: list[PlannedAttachment], pdf: Path) -> CountingEmbedder:
    embed = CountingEmbedder()
    index_plan(
        IndexPlan(collections=["Writing"], attachments=attachments),
        store=store,
        download=lambda key: pdf,
        embed=embed,
        chunk_size=1000,
        chunk_overlap=200,
    )
    return embed


def test_indexing_writes_the_keys_on_every_chunk(store: ChromaStore, pdf: Path) -> None:
    index(store, [planned(collection_keys=["K1", "K2"])], pdf)
    chunks = store.get_by_attachment("ATT1")
    assert len(chunks) == 2
    assert all(chunk.metadata["collection_keys"] == ["K1", "K2"] for chunk in chunks)


def test_a_freshly_indexed_store_needs_no_refresh(store: ChromaStore, pdf: Path) -> None:
    attachments = [planned(collection_keys=["K1"])]
    index(store, attachments, pdf)
    assert refresh_collection_keys(store, IndexPlan(attachments=attachments)) == 0


def test_moving_an_item_between_folders_updates_its_keys_without_re_embedding(
    store: ChromaStore, pdf: Path
) -> None:
    """A move does not bump the attachment version, so the incremental skip misses it."""
    index(store, [planned(collection_keys=["K1"])], pdf)
    before = {chunk.id: chunk.text for chunk in store.all_chunks()}

    moved = [planned(collection_keys=["K3"])]
    embed = index(store, moved, pdf)
    assert embed.texts == 0  # still skipped as unchanged
    assert refresh_collection_keys(store, IndexPlan(attachments=moved)) == 1

    chunks = store.all_chunks()
    assert {chunk.id: chunk.text for chunk in chunks} == before
    assert all(chunk.metadata["collection_keys"] == ["K3"] for chunk in chunks)
    assert all(chunk.metadata["title"] == "A Paper" for chunk in chunks)  # merged, not replaced


def test_chunks_written_before_the_field_existed_are_backfilled(
    store: ChromaStore, pdf: Path
) -> None:
    index(store, [planned(collection_keys=[])], pdf)
    assert store.collection_keys_by_attachment() == {"ATT1": None}

    assert refresh_collection_keys(store, IndexPlan(attachments=[planned()])) == 1
    assert store.collection_keys_by_attachment() == {"ATT1": ["K1"]}


def test_chunks_that_disagree_are_rewritten(store: ChromaStore, pdf: Path) -> None:
    """For example after a backfill that was interrupted halfway through one attachment."""
    index(store, [planned(collection_keys=["K1"])], pdf)
    first = store.get_by_attachment("ATT1")[0]
    store.upsert(
        ids=[first.id],
        embeddings=[[1.0, 1.0, 1.0]],
        documents=[first.text],
        metadatas=[{**first.metadata, "collection_keys": ["K9"]}],
    )
    assert store.collection_keys_by_attachment() == {"ATT1": None}
    assert refresh_collection_keys(store, IndexPlan(attachments=[planned()])) == 1


def test_key_order_is_not_a_difference(store: ChromaStore, pdf: Path) -> None:
    index(store, [planned(collection_keys=["K2", "K1"])], pdf)
    plan = IndexPlan(attachments=[planned(collection_keys=["K1", "K2"])])
    assert refresh_collection_keys(store, plan) == 0


def test_attachments_not_in_the_store_are_left_for_indexing(store: ChromaStore) -> None:
    assert refresh_collection_keys(store, IndexPlan(attachments=[planned()])) == 0


def test_a_plan_without_membership_never_erases_stored_keys(store: ChromaStore, pdf: Path) -> None:
    index(store, [planned(collection_keys=["K1"])], pdf)
    assert refresh_collection_keys(store, IndexPlan(attachments=[planned(collection_keys=[])])) == 0
    assert store.collection_keys_by_attachment() == {"ATT1": ["K1"]}


def test_an_empty_key_list_is_refused_before_reaching_chroma(store: ChromaStore) -> None:
    with pytest.raises(ValueError, match="empty"):
        store.set_collection_keys("ATT1", [])
