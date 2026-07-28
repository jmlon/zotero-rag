"""M5 selection: which of an item's attachments gets indexed, and how it is cited.

The case that drives the design: in the target library ~100 items hold both a PDF
and an HTML snapshot of the same article, so indexing everything readable would
embed that content twice and let the copies compete in retrieval (PRD §11.2).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from zotero_rag.documents import DocumentFormat, LocatorKind
from zotero_rag.indexer import index_plan
from zotero_rag.planning import (
    DEFAULT_PREFERENCE,
    SkipReason,
    build_plan,
    resolve_preference,
)
from zotero_rag.querying import Source, collect_sources
from zotero_rag.vectorstore import StoredChunk, VectorStore

FileFactory = Callable[[str, bytes | str], Path]

CONTENT_TYPES = {
    DocumentFormat.PDF: "application/pdf",
    DocumentFormat.HTML: "text/html",
    DocumentFormat.TEXT: "text/plain",
    DocumentFormat.DOCX: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
}


def item(key: str = "ITEM1", title: str = "A Paper") -> dict[str, Any]:
    return {"key": key, "data": {"key": key, "itemType": "journalArticle", "title": title}}


def attachment(key: str, fmt: DocumentFormat, parent: str = "ITEM1") -> dict[str, Any]:
    return {
        "key": key,
        "data": {
            "key": key,
            "itemType": "attachment",
            "title": f"{key} file",
            "contentType": CONTENT_TYPES[fmt],
            "parentItem": parent,
            "linkMode": "imported_file",
            "version": 5,
        },
    }


# --- preference resolution ------------------------------------------------


def test_configured_order_is_honoured() -> None:
    order = resolve_preference(["html", "pdf"])
    assert order[0] is DocumentFormat.HTML
    assert order[1] is DocumentFormat.PDF


def test_omitted_formats_rank_last_rather_than_being_dropped() -> None:
    """Losing content because a config list was short would be a poor surprise."""
    order = resolve_preference(["html"])
    assert order[0] is DocumentFormat.HTML
    assert set(DEFAULT_PREFERENCE).issubset(set(order))


def test_unknown_and_duplicate_entries_are_ignored() -> None:
    order = resolve_preference(["pdf", "pdf", "wordperfect", "note"])
    assert order.count(DocumentFormat.PDF) == 1
    # `note` is not a competing format: notes are indexed alongside, never instead.
    assert DocumentFormat.NOTE not in order


def test_no_configuration_uses_the_default_order() -> None:
    assert resolve_preference(None) == DEFAULT_PREFERENCE
    assert resolve_preference([]) == DEFAULT_PREFERENCE


# --- selection ------------------------------------------------------------


def test_a_pdf_beats_a_snapshot_of_the_same_item() -> None:
    """The ~100-duplicate case: the published PDF wins, the snapshot is recorded."""
    plan = build_plan(
        [
            (
                "C",
                [
                    item(),
                    attachment("SNAP", DocumentFormat.HTML),
                    attachment("PDF1", DocumentFormat.PDF),
                ],
            )
        ]
    )

    assert [a.attachment_key for a in plan.attachments] == ["PDF1"]
    rejected = plan.skipped_for(SkipReason.NOT_PREFERRED)
    assert [entry.key for entry in rejected] == ["SNAP"]
    assert plan.content_types_for(SkipReason.NOT_PREFERRED) == {"text/html": 1}


def test_preference_can_be_inverted_by_configuration() -> None:
    plan = build_plan(
        [
            (
                "C",
                [
                    item(),
                    attachment("SNAP", DocumentFormat.HTML),
                    attachment("PDF1", DocumentFormat.PDF),
                ],
            )
        ],
        preference=["html", "pdf"],
    )
    assert [a.attachment_key for a in plan.attachments] == ["SNAP"]


def test_index_all_attachments_keeps_every_readable_copy() -> None:
    plan = build_plan(
        [
            (
                "C",
                [
                    item(),
                    attachment("SNAP", DocumentFormat.HTML),
                    attachment("PDF1", DocumentFormat.PDF),
                ],
            )
        ],
        index_all_attachments=True,
    )
    assert {a.attachment_key for a in plan.attachments} == {"SNAP", "PDF1"}
    assert plan.skipped_for(SkipReason.NOT_PREFERRED) == []


def test_two_attachments_of_the_same_format_are_both_indexed() -> None:
    """PRD §7.1 still holds: a paper and its supplement are different content."""
    plan = build_plan(
        [
            (
                "C",
                [
                    item(),
                    attachment("MAIN", DocumentFormat.PDF),
                    attachment("SUPP", DocumentFormat.PDF),
                ],
            )
        ]
    )
    assert {a.attachment_key for a in plan.attachments} == {"MAIN", "SUPP"}
    assert plan.skipped_for(SkipReason.NOT_PREFERRED) == []


def test_preference_applies_per_item_not_across_the_collection() -> None:
    """One item having a PDF must not suppress another item's only snapshot."""
    plan = build_plan(
        [
            (
                "C",
                [
                    item("ITEM1", "Has a PDF"),
                    attachment("PDF1", DocumentFormat.PDF, parent="ITEM1"),
                    item("ITEM2", "Snapshot only"),
                    attachment("SNAP2", DocumentFormat.HTML, parent="ITEM2"),
                ],
            )
        ]
    )
    assert {a.attachment_key for a in plan.attachments} == {"PDF1", "SNAP2"}
    assert plan.format_counts == {DocumentFormat.PDF: 1, DocumentFormat.HTML: 1}


def test_format_counts_summarise_a_mixed_plan() -> None:
    plan = build_plan(
        [
            (
                "C",
                [
                    item("ITEM1"),
                    attachment("A", DocumentFormat.PDF, parent="ITEM1"),
                    item("ITEM2"),
                    attachment("B", DocumentFormat.HTML, parent="ITEM2"),
                    item("ITEM3"),
                    attachment("C", DocumentFormat.TEXT, parent="ITEM3"),
                ],
            )
        ]
    )
    assert plan.format_counts[DocumentFormat.PDF] == 1
    assert plan.format_counts[DocumentFormat.HTML] == 1
    assert plan.format_counts[DocumentFormat.TEXT] == 1


# --- end to end through the indexer --------------------------------------


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(text)), 1.0, 0.5] for text in texts]


def test_a_snapshot_is_indexed_with_section_locators(
    tmp_path: Path, file_factory: FileFactory
) -> None:
    """The whole point of M5: snapshot-only items become answerable."""
    snapshot = file_factory(
        "snap.html",
        "<html><body><h1>What is Fog Computing?</h1><p>A definition follows.</p>"
        "<h2>Architecture</h2><p>Nodes at the edge.</p></body></html>",
    )
    plan = build_plan([("C", [item(), attachment("SNAP", DocumentFormat.HTML)])])
    store = VectorStore(tmp_path / "chroma")

    stats = index_plan(
        plan,
        store=store,
        download=lambda key: snapshot,
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="m1",
    )

    assert stats.indexed == 1
    chunks = store.get_by_attachment("SNAP")
    assert len(chunks) == 2
    kinds = {chunk.metadata["locator_kind"] for chunk in chunks}
    assert kinds == {str(LocatorKind.SECTION)}
    labels = {chunk.metadata["locator_label"] for chunk in chunks}
    assert labels == {"What is Fog Computing?", "Architecture"}
    assert {chunk.metadata["document_format"] for chunk in chunks} == {"html"}


def test_a_note_is_indexed_without_any_download(tmp_path: Path) -> None:
    note = {
        "key": "NOTE1",
        "data": {
            "key": "NOTE1",
            "itemType": "note",
            "note": "<p>This paper matters because of X.</p>",
            "parentItem": "ITEM1",
            "version": 3,
        },
    }
    plan = build_plan([("C", [item(), note])])
    store = VectorStore(tmp_path / "chroma")

    def refuse_download(key: str) -> Path:
        raise AssertionError(f"a note must not be downloaded (asked for {key})")

    stats = index_plan(
        plan,
        store=store,
        download=refuse_download,
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="m1",
    )

    assert stats.indexed == 1
    chunk = store.get_by_attachment("NOTE1")[0]
    assert "This paper matters because of X." in chunk.text
    assert chunk.metadata["locator_kind"] == str(LocatorKind.NOTE)
    # It carries the parent item's citation metadata, not the note's own.
    assert chunk.metadata["title"] == "A Paper"


# --- citation rendering ---------------------------------------------------


def stored(metadata: dict[str, Any]) -> StoredChunk:
    return StoredChunk(id="K:1:0", text="text", metadata={"attachment_key": "K", **metadata})


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"page": 4, "locator_kind": "page"}, "p. 4"),
        ({"page": 2, "locator_kind": "section", "locator_label": "Methods"}, "§ Methods"),
        ({"page": 7, "locator_kind": "section"}, "§ 7"),
        ({"page": 12, "locator_kind": "paragraph"}, "¶ 12"),
        ({"page": 1, "locator_kind": "note"}, "note"),
        ({"page": 3, "locator_kind": "chunk"}, "chunk 3"),
    ],
)
def test_each_locator_kind_is_cited_in_its_own_terms(
    metadata: dict[str, Any], expected: str
) -> None:
    sources, _ = collect_sources([stored(metadata)])
    assert sources[0].locator == expected
    assert sources[0].label.endswith(expected)


def test_chunks_written_before_m5_are_still_cited_as_pages() -> None:
    """No locator_kind means a PDF page, because that is all there used to be."""
    sources, _ = collect_sources([stored({"page": 9})])
    assert sources[0].locator == "p. 9"


def test_a_full_citation_reads_naturally_for_a_snapshot() -> None:
    source = Source(
        number=1,
        title="What is a Fog Node",
        authors="Tordera, Eva Marin",
        year="2016",
        page=2,
        locator_kind="section",
        locator_label="Architecture",
    )
    assert source.label == "What is a Fog Node — Tordera, Eva Marin (2016), § Architecture"
