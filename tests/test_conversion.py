"""Phase 3 of PLAN-markdown: which attachments need a transcript, and what to do.

Payloads are shaped like real Web API items, including the provenance note a
transcript carries. Keys are eight characters of [A-Z0-9], as Zotero's are,
because the provenance parser checks that shape.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from zotero_rag.conversion import (
    ConversionAction,
    ConversionState,
    build_conversion_plan,
)
from zotero_rag_core.planning import SkipReason
from zotero_rag_core.provenance import Provenance

PDF_MD5 = "a" * 32
NEW_PDF_MD5 = "b" * 32
MD_MD5 = "c" * 32
CONVERTER = "anydoc 0.2.4"


def paper(key: str = "ITEM0001", title: str = "A Paper") -> dict[str, Any]:
    data = {
        "key": key,
        "itemType": "journalArticle",
        "title": title,
        "creators": [{"creatorType": "author", "lastName": "Doe", "firstName": "Jane"}],
        "date": "2021-06",
        "DOI": "10.1/x",
    }
    return {"key": key, "data": data}


def attachment(
    key: str,
    parent: str = "ITEM0001",
    content_type: str = "application/pdf",
    *,
    md5: str | None = PDF_MD5,
    link_mode: str = "imported_file",
    filename: str = "paper.pdf",
    note: str = "",
    version: int = 10,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "key": key,
        "itemType": "attachment",
        "parentItem": parent,
        "title": filename,
        "contentType": content_type,
        "linkMode": link_mode,
        "filename": filename,
        "md5": md5,
        "note": note,
        "version": version,
    }
    return {"key": key, "data": data}


def transcript(
    key: str = "MDKEY001",
    *,
    source: str = "PDFKEY01",
    source_md5: str = PDF_MD5,
    recorded_md5: str = MD_MD5,
    zotero_md5: str = MD_MD5,
    converter: str = CONVERTER,
    parent: str = "ITEM0001",
    version: int = 20,
    **extra: Any,
) -> dict[str, Any]:
    record = Provenance(
        source=source, source_md5=source_md5, md5=recorded_md5, converter=converter, **extra
    )
    return attachment(
        key,
        parent,
        "text/markdown",
        md5=zotero_md5,
        filename="paper.md",
        note=record.to_note(),
        version=version,
    )


def plan_for(*items: dict[str, Any], **options: Any) -> Any:
    options.setdefault("converter_for", lambda source_format: CONVERTER)
    return build_conversion_plan([("Library", list(items))], **options)


def only_entry(plan: Any) -> Any:
    assert len(plan.entries) == 1, plan.entries
    return plan.entries[0]


# --- The states of a source -------------------------------------------------------


def test_a_pdf_without_a_transcript_is_created() -> None:
    plan = plan_for(paper(), attachment("PDFKEY01"))

    entry = only_entry(plan)
    assert entry.state is ConversionState.MISSING
    assert entry.action is ConversionAction.CREATE
    assert entry.source.attachment_key == "PDFKEY01"
    assert entry.source.source_format == "pdf"
    assert entry.source.md5 == PDF_MD5
    assert entry.source.authors.startswith("Doe")
    assert entry.source.year == "2021"
    assert plan.actions == [entry]


def test_a_current_transcript_is_left_alone() -> None:
    entry = only_entry(plan_for(paper(), attachment("PDFKEY01"), transcript()))

    assert entry.state is ConversionState.CURRENT
    assert entry.action is ConversionAction.NONE
    assert entry.transcript.key == "MDKEY001"


def test_a_replaced_source_is_regenerated_in_place() -> None:
    plan = plan_for(paper(), attachment("PDFKEY01", md5=NEW_PDF_MD5), transcript())

    entry = only_entry(plan)
    assert entry.state is ConversionState.STALE
    assert entry.action is ConversionAction.REGENERATE
    assert entry.transcript.key == "MDKEY001"  # replaced, not duplicated
    assert entry.transcript.md5 == MD_MD5  # the precondition for the replace


def test_an_edited_transcript_is_kept_unless_forced() -> None:
    items = (paper(), attachment("PDFKEY01"), transcript(zotero_md5="d" * 32))

    kept = only_entry(plan_for(*items))
    forced = only_entry(plan_for(*items, force=True))

    assert kept.state is forced.state is ConversionState.EDITED
    assert kept.action is ConversionAction.NONE
    assert forced.action is ConversionAction.REGENERATE


def test_an_edit_outranks_a_changed_source() -> None:
    # Regenerating would throw away a person's corrections, so the edit wins.
    entry = only_entry(
        plan_for(
            paper(),
            attachment("PDFKEY01", md5=NEW_PDF_MD5),
            transcript(zotero_md5="d" * 32),
        )
    )

    assert entry.state is ConversionState.EDITED
    assert entry.action is ConversionAction.NONE


def test_an_older_converter_regenerates_only_on_request() -> None:
    items = (paper(), attachment("PDFKEY01"), transcript(converter="anydoc 0.2.3"))

    assert only_entry(plan_for(*items)).action is ConversionAction.NONE
    again = only_entry(plan_for(*items, reconvert=True))
    assert again.state is ConversionState.OUTDATED
    assert again.action is ConversionAction.REGENERATE


def test_markdown_the_user_attached_counts_as_converted() -> None:
    plan = plan_for(
        paper(),
        attachment("PDFKEY01"),
        attachment("USERMD01", content_type="text/markdown", filename="my notes.md"),
        force=True,
    )

    entry = only_entry(plan)
    assert entry.state is ConversionState.USER_PROVIDED
    assert entry.action is ConversionAction.NONE  # --force does not override it
    assert [md.key for md in entry.user_markdown] == ["USERMD01"]
    assert plan.actions == []


def test_a_user_md_stored_as_plain_text_is_recognised_by_name() -> None:
    entry = only_entry(
        plan_for(
            paper(),
            attachment("PDFKEY01"),
            attachment("USERMD01", content_type="text/plain", filename="Transcript.MD"),
        )
    )

    assert entry.state is ConversionState.USER_PROVIDED


def test_a_linked_user_md_still_counts() -> None:
    entry = only_entry(
        plan_for(
            paper(),
            attachment("PDFKEY01"),
            attachment(
                "USERMD01",
                content_type="text/markdown",
                link_mode="linked_file",
                md5=None,
                filename="notes.md",
            ),
        )
    )

    assert entry.state is ConversionState.USER_PROVIDED
    assert entry.user_markdown[0].link_mode == "linked_file"


def test_a_user_md_outranks_our_own_stale_transcript() -> None:
    entry = only_entry(
        plan_for(
            paper(),
            attachment("PDFKEY01", md5=NEW_PDF_MD5),
            transcript(),
            attachment("USERMD01", content_type="text/markdown", filename="mine.md"),
        )
    )

    assert entry.state is ConversionState.USER_PROVIDED
    assert entry.transcript.key == "MDKEY001"
    assert entry.action is ConversionAction.NONE


def test_a_markdown_note_that_is_not_ours_is_a_user_md() -> None:
    # Someone else's note on a .md file: the transcript is theirs, not ours.
    foreign = attachment(
        "USERMD01", content_type="text/markdown", filename="x.md", note="my reading notes"
    )

    entry = only_entry(plan_for(paper(), attachment("PDFKEY01"), foreign))

    assert entry.state is ConversionState.USER_PROVIDED


# --- Choosing the source ----------------------------------------------------------


def test_one_transcript_per_item_from_the_preferred_format() -> None:
    plan = plan_for(
        paper(),
        attachment("PDFKEY01"),
        attachment("SNAPSHT1", content_type="text/html", link_mode="imported_url"),
    )

    entry = only_entry(plan)
    assert entry.source.attachment_key == "PDFKEY01"
    assert [s.key for s in plan.skipped_for(SkipReason.NOT_PREFERRED)] == ["SNAPSHT1"]


def test_preference_can_put_snapshots_first() -> None:
    plan = plan_for(
        paper(),
        attachment("PDFKEY01"),
        attachment("SNAPSHT1", content_type="text/html", link_mode="imported_url"),
        preference=["html", "pdf"],
    )

    assert only_entry(plan).source.attachment_key == "SNAPSHT1"


def test_every_attachment_converts_all_formats() -> None:
    plan = plan_for(
        paper(),
        attachment("PDFKEY01"),
        attachment("SNAPSHT1", content_type="text/html", link_mode="imported_url"),
        every_attachment=True,
    )

    assert sorted(e.source.attachment_key for e in plan.entries) == ["PDFKEY01", "SNAPSHT1"]


def test_a_paper_and_its_supplement_each_get_a_transcript() -> None:
    plan = plan_for(paper(), attachment("PDFKEY01"), attachment("PDFKEY02"))

    assert len(plan.actions) == 2


def test_formats_only_conversion_reads_are_sources() -> None:
    slides = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

    entry = only_entry(plan_for(paper(), attachment("SLIDES01", content_type=slides)))

    assert entry.source.source_format == "pptx"
    assert entry.source.document_format is None  # index does not read pptx
    assert entry.action is ConversionAction.CREATE


def test_text_attachments_need_no_transcript() -> None:
    plan = plan_for(paper(), attachment("TXTKEY01", content_type="text/plain", filename="a.txt"))

    assert plan.entries == []
    assert [s.key for s in plan.skipped_for(SkipReason.ALREADY_TEXT)] == ["TXTKEY01"]


def test_unreachable_files_are_reported_as_index_reports_them() -> None:
    plan = plan_for(
        paper(),
        attachment("LINKED01", link_mode="linked_file", md5=None),
        paper("ITEM0002"),
        attachment("UNSYNC01", parent="ITEM0002", md5=None),
        paper("ITEM0003"),
        attachment("LINKURL1", parent="ITEM0003", link_mode="linked_url", content_type=""),
    )

    assert plan.entries == []
    assert plan.skip_counts[SkipReason.LINKED_FILE] == 1
    assert plan.skip_counts[SkipReason.NOT_IN_STORAGE] == 1
    assert plan.skip_counts[SkipReason.LINKED_URL] == 1
    # The unsynced items are reported under their fix, not also as "no content".
    assert [s.key for s in plan.skipped_for(SkipReason.NO_CONTENT)] == ["ITEM0003"]


def test_an_item_filed_twice_is_planned_once() -> None:
    items = [paper(), attachment("PDFKEY01"), transcript()]
    plan = build_conversion_plan(
        [("One", items), ("Two", items)], converter_for=lambda source_format: CONVERTER
    )

    assert len(plan.entries) == 1
    assert plan.items_seen == 1
    assert plan.skip_counts == {SkipReason.DUPLICATE: 1}


def test_a_repeated_text_file_is_a_duplicate_not_unsupported() -> None:
    items = [paper(), attachment("TXTKEY01", content_type="text/plain", filename="a.txt")]
    plan = build_conversion_plan([("One", items), ("Two", items)])

    assert plan.skip_counts == {SkipReason.ALREADY_TEXT: 1}


# --- Transcripts no source claims -------------------------------------------------


def test_a_transcript_whose_source_was_shrunk_is_never_regenerated() -> None:
    gone = transcript(source_removed=date(2026, 11, 2), source_url="https://x.org/a.pdf")

    plan = plan_for(paper(), gone)

    entry = only_entry(plan)
    assert entry.state is ConversionState.SOURCE_GONE
    assert entry.action is ConversionAction.NONE
    assert entry.title == "A Paper"
    # The paper has its transcript; it is not "without readable content".
    assert plan.skipped == []


def test_a_transcript_whose_source_vanished_is_reported_not_deleted() -> None:
    entry = only_entry(plan_for(paper(), transcript()))

    assert entry.state is ConversionState.SOURCE_MISSING
    assert entry.action is ConversionAction.NONE


def test_a_transcript_of_a_no_longer_preferred_source_is_superseded() -> None:
    # The item had only a snapshot when it was converted; a PDF arrived since.
    plan = plan_for(
        paper(),
        attachment("PDFKEY01"),
        attachment("SNAPSHT1", content_type="text/html", link_mode="imported_url"),
        transcript(source="SNAPSHT1"),
    )

    states = {entry.state for entry in plan.entries}
    assert states == {ConversionState.MISSING, ConversionState.SUPERSEDED}
    assert [entry.source.attachment_key for entry in plan.actions] == ["PDFKEY01"]


def test_a_second_transcript_of_one_source_is_a_reported_duplicate() -> None:
    plan = plan_for(
        paper(),
        attachment("PDFKEY01"),
        transcript("MDKEY001", version=20),
        transcript("MDKEY002", version=30),
    )

    by_state = {entry.state: entry for entry in plan.entries}
    assert by_state[ConversionState.CURRENT].transcript.key == "MDKEY002"
    assert by_state[ConversionState.DUPLICATE].transcript.key == "MDKEY001"
    assert plan.actions == []


def test_a_transcript_outside_the_target_is_ignored() -> None:
    plan = plan_for(paper(), attachment("PDFKEY01"), transcript(parent="ELSEWHER"))

    assert only_entry(plan).state is ConversionState.MISSING
    assert plan.skipped == []


def test_state_counts_summarise_the_plan() -> None:
    plan = plan_for(
        paper(),
        attachment("PDFKEY01"),
        transcript(),
        paper("ITEM0002"),
        attachment("PDFKEY02", parent="ITEM0002"),
    )

    assert plan.state_counts == {ConversionState.CURRENT: 1, ConversionState.MISSING: 1}
