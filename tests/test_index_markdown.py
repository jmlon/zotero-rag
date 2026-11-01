"""Phase 6 of PLAN-markdown: indexing Markdown, and transcripts as copies of a source.

A transcript made by `convert` and its source hold the same text. These tests pin
the rule that exactly one of them is indexed, chosen by `attachment_preference`,
and that switching the preference swaps the copy without leaving the old one's
chunks behind.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from zotero_rag_core.documents import DocumentFormat, LocatorKind
from zotero_rag_core.extraction import extract_segments
from zotero_rag_core.indexer import drop_replaced_copies, find_orphans
from zotero_rag_core.planning import SkipReason, build_plan
from zotero_rag_core.provenance import Provenance, parse
from zotero_rag_core.stores.chroma import ChromaStore

FileFactory = Callable[[str, bytes | str], Path]
MARKDOWN_FIRST = ["markdown", "pdf", "docx", "odt", "epub", "html", "txt"]
PDF_FIRST = ["pdf", "docx", "odt", "epub", "html", "markdown", "txt"]
# What `init` wrote before Markdown was a format of its own.
OLD_FOLDER = ["pdf", "docx", "odt", "epub", "html", "txt"]


# --- Reading Markdown ---------------------------------------------------------------


def test_transcript_pages_become_page_segments(file_factory: FileFactory) -> None:
    path = file_factory(
        "t.md",
        "---\ntitle: A Paper\nauthors: Doe\n---\n\n"
        "<!-- page 1 -->\n\n# Introduction\n\nFirst page.\n\n"
        "<!-- page 2 -->\n\n<!-- page 3 -->\n\nThird page.\n",
    )

    segments = extract_segments(path, DocumentFormat.MARKDOWN)

    assert [(s.number, s.kind) for s in segments] == [(1, LocatorKind.PAGE), (3, LocatorKind.PAGE)]
    assert segments[0].text == "# Introduction\n\nFirst page."
    # Front matter describes the file; it is not text to retrieve.
    assert all("authors" not in s.text for s in segments)


def test_markdown_without_markers_splits_on_headings(file_factory: FileFactory) -> None:
    path = file_factory("notes.md", "# Methods\n\nWhat we did.\n\n## Results\n\nWhat we found.\n")

    segments = extract_segments(path, DocumentFormat.MARKDOWN)

    assert [(s.label, s.kind) for s in segments] == [
        ("Methods", LocatorKind.SECTION),
        ("Results", LocatorKind.SECTION),
    ]


def test_provenance_records_the_source_format() -> None:
    record = Provenance(
        source="PDFKEY01", source_md5="a" * 32, md5="b" * 32, converter="x", source_format="pdf"
    )

    assert parse(record.to_note()) == record
    assert parse(record.to_note().replace("\nsource_format=pdf", "")).source_format == ""  # type: ignore[union-attr]


# --- Choosing one copy --------------------------------------------------------------


def paper(key: str = "ITEM0001") -> dict[str, Any]:
    return {"key": key, "data": {"key": key, "itemType": "journalArticle", "title": "A Paper"}}


def attachment(
    key: str, content_type: str = "application/pdf", parent: str = "ITEM0001", **extra: Any
) -> dict[str, Any]:
    data = {
        "key": key,
        "itemType": "attachment",
        "parentItem": parent,
        "contentType": content_type,
        "linkMode": "imported_file",
        "md5": "a" * 32,
        "title": key,
        "version": 5,
    }
    data.update(extra)
    return {"key": key, "data": data}


def transcript_of(
    source: str, key: str = "MDKEY001", source_format: str = "pdf", **extra: Any
) -> dict[str, Any]:
    note = Provenance(
        source=source,
        source_md5="a" * 32,
        md5="b" * 32,
        converter="anydoc 0.2.4",
        source_format=source_format,
    ).to_note()
    return attachment(key, "text/markdown", note=note, **extra)


def plan_of(*items: dict[str, Any], **options: Any) -> Any:
    return build_plan([("Library", list(items))], **options)


def indexed(plan: Any) -> list[str]:
    return sorted(attachment.attachment_key for attachment in plan.attachments)


def test_by_default_the_transcript_is_indexed_instead_of_its_pdf() -> None:
    plan = plan_of(paper(), attachment("PDFKEY01"), transcript_of("PDFKEY01"))

    assert indexed(plan) == ["MDKEY001"]
    (skip,) = plan.skipped_for(SkipReason.SAME_DOCUMENT)
    assert (skip.key, skip.replaced_by) == ("PDFKEY01", "MDKEY001")


def test_config_can_put_the_originals_first() -> None:
    plan = plan_of(paper(), attachment("PDFKEY01"), transcript_of("PDFKEY01"), preference=PDF_FIRST)

    assert indexed(plan) == ["PDFKEY01"]
    (skip,) = plan.skipped_for(SkipReason.SAME_DOCUMENT)
    assert (skip.key, skip.replaced_by) == ("MDKEY001", "PDFKEY01")


def test_a_folder_made_before_markdown_existed_keeps_its_originals() -> None:
    # Its list never mentions markdown, which therefore ranks last.
    plan = plan_of(
        paper(), attachment("PDFKEY01"), transcript_of("PDFKEY01"), preference=OLD_FOLDER
    )

    assert indexed(plan) == ["PDFKEY01"]


def test_without_a_transcript_the_default_still_prefers_the_pdf() -> None:
    plan = plan_of(paper(), attachment("PDFKEY01"), attachment("SNAPSHT1", "text/html"))

    assert indexed(plan) == ["PDFKEY01"]


def test_markdown_first_indexes_the_transcript_instead() -> None:
    plan = plan_of(
        paper(), attachment("PDFKEY01"), transcript_of("PDFKEY01"), preference=MARKDOWN_FIRST
    )

    assert indexed(plan) == ["MDKEY001"]
    assert plan.attachments[0].document_format is DocumentFormat.MARKDOWN
    (skip,) = plan.skipped_for(SkipReason.SAME_DOCUMENT)
    assert (skip.key, skip.replaced_by) == ("PDFKEY01", "MDKEY001")


def test_never_both_copies_even_when_indexing_every_attachment() -> None:
    plan = plan_of(
        paper(),
        attachment("PDFKEY01"),
        transcript_of("PDFKEY01"),
        index_all_attachments=True,
    )

    assert indexed(plan) == ["MDKEY001"]


def test_a_transcript_does_not_push_out_a_supplement() -> None:
    # The format choice is between documents: the paper (PDF, via its
    # transcript) and its supplement (PDF) are both PDFs, so both stay.
    plan = plan_of(
        paper(),
        attachment("PDFKEY01"),
        transcript_of("PDFKEY01"),
        attachment("SUPPLEM1"),
        preference=MARKDOWN_FIRST,
    )

    assert indexed(plan) == ["MDKEY001", "SUPPLEM1"]


def test_a_transcript_of_a_less_preferred_source_follows_its_source_out() -> None:
    plan = plan_of(
        paper(),
        attachment("PDFKEY01"),
        attachment("SNAPSHT1", "text/html"),
        transcript_of("SNAPSHT1", source_format="html"),
        preference=MARKDOWN_FIRST,
    )

    assert indexed(plan) == ["PDFKEY01"]
    assert sorted(s.key for s in plan.skipped_for(SkipReason.NOT_PREFERRED)) == [
        "MDKEY001",
        "SNAPSHT1",
    ]


def test_a_transcript_whose_pdf_was_shrunk_keeps_the_pdfs_standing() -> None:
    # The PDF went to the trash; its transcript must still beat the snapshot.
    plan = plan_of(
        paper(),
        attachment("SNAPSHT1", "text/html"),
        transcript_of("TRASHED1", source_format="pdf"),
    )

    assert indexed(plan) == ["MDKEY001"]
    assert plan.attachments[0].document_format is DocumentFormat.MARKDOWN


def test_a_second_transcript_of_one_source_is_never_indexed() -> None:
    plan = plan_of(
        paper(),
        attachment("PDFKEY01"),
        transcript_of("PDFKEY01", "MDKEY001", version=5),
        transcript_of("PDFKEY01", "MDKEY002", version=9),
        preference=MARKDOWN_FIRST,
    )

    assert indexed(plan) == ["MDKEY002"]
    assert sorted(s.key for s in plan.skipped_for(SkipReason.SAME_DOCUMENT)) == [
        "MDKEY001",
        "PDFKEY01",
    ]


def test_markdown_the_user_wrote_is_an_ordinary_document() -> None:
    plan = plan_of(paper(), attachment("USERMD01", "text/markdown", note="my notes"))

    assert indexed(plan) == ["USERMD01"]
    assert plan.skipped == []


# --- Swapping copies in the store ---------------------------------------------------


def stored(store: ChromaStore, *keys: str) -> None:
    for number, key in enumerate(keys):
        store.add(
            ids=[f"{key}:1:0"],
            embeddings=[[float(number), 1.0, 0.5]],
            documents=[f"text of {key}"],
            metadatas=[{"attachment_key": key, "item_key": "ITEM0001", "page": 1}],
        )


def test_switching_to_the_transcript_drops_the_pdfs_chunks(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    stored(store, "PDFKEY01", "MDKEY001")
    plan = plan_of(
        paper(), attachment("PDFKEY01"), transcript_of("PDFKEY01"), preference=MARKDOWN_FIRST
    )

    assert drop_replaced_copies(store, plan) == (1, 1)
    assert store.attachment_keys() == {"MDKEY001"}
    assert find_orphans(store, plan) == set()


def test_the_old_copy_stays_until_the_new_one_is_stored(tmp_path: Path) -> None:
    # The transcript failed to index this run: dropping the PDF would lose the
    # paper from the index entirely.
    store = ChromaStore(tmp_path / "chroma")
    stored(store, "PDFKEY01")
    plan = plan_of(
        paper(), attachment("PDFKEY01"), transcript_of("PDFKEY01"), preference=MARKDOWN_FIRST
    )

    assert drop_replaced_copies(store, plan) == (0, 0)
    assert store.attachment_keys() == {"PDFKEY01"}
