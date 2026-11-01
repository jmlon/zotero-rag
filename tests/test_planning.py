"""Selection rules from PRD §7.1, applied to raw Zotero item payloads."""

from __future__ import annotations

from typing import Any

from zotero_rag_core.documents import DocumentFormat
from zotero_rag_core.planning import SkipReason, build_plan, item_membership


def item(
    key: str,
    title: str = "A Paper",
    item_type: str = "journalArticle",
    **extra: Any,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "key": key,
        "itemType": item_type,
        "title": title,
        "creators": [{"creatorType": "author", "lastName": "Doe", "firstName": "Jane"}],
        "date": "2021-06",
    }
    data.update(extra)
    return {"key": key, "data": data}


def attachment(
    key: str,
    parent: str | None = "ITEM1",
    content_type: str = "application/pdf",
    link_mode: str = "imported_file",
    title: str = "Full Text PDF",
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "key": key,
        "itemType": "attachment",
        "title": title,
        "contentType": content_type,
        "linkMode": link_mode,
    }
    if parent is not None:
        data["parentItem"] = parent
    return {"key": key, "data": data}


def test_pdf_attachment_is_planned_with_parent_metadata() -> None:
    plan = build_plan([("Writting", [item("ITEM1", "Writing Science"), attachment("ATT1")])])

    assert len(plan.attachments) == 1
    planned = plan.attachments[0]
    assert planned.attachment_key == "ATT1"
    assert planned.item_key == "ITEM1"
    assert planned.title == "Writing Science"
    assert planned.authors == "Doe, Jane"
    assert planned.year == "2021"
    assert planned.collection_name == "Writting"
    assert plan.items_seen == 1
    assert plan.item_count == 1
    assert plan.skipped == []


def test_multiple_pdfs_on_one_item_are_all_planned() -> None:
    plan = build_plan([("C", [item("ITEM1"), attachment("ATT1"), attachment("ATT2")])])
    assert {a.attachment_key for a in plan.attachments} == {"ATT1", "ATT2"}
    # Two attachments, but a single source item.
    assert plan.item_count == 1


def test_linked_url_attachment_is_skipped() -> None:
    plan = build_plan(
        [
            (
                "C",
                [
                    item("ITEM1"),
                    attachment("ATT1", content_type="text/html", link_mode="linked_url"),
                ],
            )
        ]
    )
    assert plan.attachments == []
    reasons = {entry.reason for entry in plan.skipped}
    assert SkipReason.LINKED_URL in reasons
    # The parent is also reported as having no usable PDF.
    assert SkipReason.NO_CONTENT in reasons


def test_a_snapshot_is_indexed_when_it_is_all_the_item_has() -> None:
    """The M5 payoff: 105 items in the target library have only a snapshot."""
    plan = build_plan([("C", [item("ITEM1"), attachment("ATT1", content_type="text/html")])])
    assert [a.document_format for a in plan.attachments] == [DocumentFormat.HTML]
    assert plan.skipped == []


def test_a_genuinely_unreadable_type_is_still_skipped_with_its_type() -> None:
    plan = build_plan([("C", [item("ITEM1"), attachment("ATT1", content_type="image/png")])])
    assert plan.attachments == []
    unsupported = plan.skipped_for(SkipReason.UNSUPPORTED_TYPE)
    assert [e.content_type for e in unsupported] == ["image/png"]
    assert plan.content_types_for(SkipReason.UNSUPPORTED_TYPE) == {"image/png": 1}


def test_epub_is_indexed() -> None:
    plan = build_plan(
        [("C", [item("ITEM1"), attachment("ATT1", content_type="application/epub+zip")])]
    )
    assert [a.document_format for a in plan.attachments] == [DocumentFormat.EPUB]


def test_item_without_any_attachment_is_reported() -> None:
    plan = build_plan([("C", [item("ITEM1", "Lonely Item")])])
    assert plan.attachments == []
    assert len(plan.skipped) == 1
    assert plan.skipped[0].reason == SkipReason.NO_CONTENT
    assert plan.skipped[0].title == "Lonely Item"


def test_notes_are_indexed_alongside_their_item() -> None:
    note = {
        "key": "NOTE1",
        "data": {
            "key": "NOTE1",
            "itemType": "note",
            "note": "Remember to cite this\nsecond line",
            "parentItem": "ITEM1",
        },
    }
    plan = build_plan([("C", [item("ITEM1"), attachment("ATT1"), note])])

    # The note is indexed *as well as* the PDF: it complements the paper.
    formats = [a.document_format for a in plan.attachments]
    assert formats == [DocumentFormat.PDF, DocumentFormat.NOTE]
    indexed_note = plan.attachments[1]
    assert indexed_note.note_html.startswith("Remember to cite this")
    assert indexed_note.needs_download is False
    # It inherits the parent's citation metadata.
    assert indexed_note.item_key == "ITEM1"
    # A note must not make its parent look item-less.
    assert plan.items_seen == 1


def test_notes_can_be_turned_off() -> None:
    note = {
        "key": "NOTE1",
        "data": {"key": "NOTE1", "itemType": "note", "note": "A remark", "parentItem": "ITEM1"},
    }
    plan = build_plan([("C", [item("ITEM1"), attachment("ATT1"), note])], index_notes=False)
    assert [a.document_format for a in plan.attachments] == [DocumentFormat.PDF]
    assert [e.reason for e in plan.skipped] == [SkipReason.NOTE]


def test_a_standalone_note_is_skipped() -> None:
    """With no parent item there is no bibliographic metadata to cite."""
    note = {"key": "NOTE1", "data": {"key": "NOTE1", "itemType": "note", "note": "Loose thought"}}
    plan = build_plan([("C", [note])])
    assert plan.attachments == []
    assert [e.reason for e in plan.skipped] == [SkipReason.NOTE]


def test_attachment_without_a_parent_is_skipped() -> None:
    plan = build_plan([("C", [attachment("ATT1", parent=None)])])
    assert plan.attachments == []
    assert plan.skipped[0].reason == SkipReason.ORPHAN_ATTACHMENT


def test_attachment_whose_parent_is_out_of_scope_is_skipped() -> None:
    plan = build_plan([("C", [attachment("ATT1", parent="ELSEWHERE")])])
    assert plan.attachments == []
    assert plan.skipped[0].reason == SkipReason.PARENT_OUT_OF_SCOPE


def test_item_shared_by_two_collections_is_embedded_once() -> None:
    """PRD §7.1: deduplicate by attachment_key across subcollections."""
    shared_item = item("ITEM1", "Shared Paper")
    shared_attachment = attachment("ATT1")
    plan = build_plan(
        [
            ("Parent", [shared_item, shared_attachment]),
            ("Child", [shared_item, shared_attachment]),
        ]
    )
    assert len(plan.attachments) == 1
    assert [entry.reason for entry in plan.skipped] == [SkipReason.DUPLICATE]


def test_parent_in_one_collection_attachment_in_another_still_resolves() -> None:
    """Parents are gathered across all collections before attachments resolve."""
    plan = build_plan([("Parent", [item("ITEM1")]), ("Child", [attachment("ATT1")])])
    assert len(plan.attachments) == 1
    assert plan.attachments[0].item_key == "ITEM1"


def test_plan_summarises_collections_and_skip_counts() -> None:
    plan = build_plan(
        [
            (
                "C",
                [
                    item("ITEM1"),
                    attachment("ATT1"),
                    item("ITEM2", "No File"),
                    item("ITEM3", "Also No File"),
                    attachment("ATT2", parent="ITEM1", content_type="text/html"),
                ],
            )
        ]
    )
    assert plan.collections == ["C"]
    assert plan.items_seen == 3
    # ITEM1 offers a PDF and a snapshot; the PDF wins and the snapshot is recorded.
    assert [a.document_format for a in plan.attachments] == [DocumentFormat.PDF]
    counts = plan.skip_counts
    assert counts[SkipReason.NO_CONTENT] == 2
    assert counts[SkipReason.NOT_PREFERRED] == 1


def test_entries_without_keys_are_ignored() -> None:
    plan = build_plan([("C", [{"data": {"itemType": "journalArticle", "title": "Broken"}}])])
    assert plan.attachments == []
    assert plan.items_seen == 0


def test_empty_collection_yields_an_empty_plan() -> None:
    plan = build_plan([("Empty", [])])
    assert plan.attachments == []
    assert plan.skipped == []
    assert plan.items_seen == 0
    assert plan.item_count == 0


# --- collection membership (PRD-CursesUI §4.3) ----------------------------


def test_membership_lists_every_fetched_collection_holding_the_item() -> None:
    membership = item_membership(
        [
            ("KPARENT", [item("ITEM1"), attachment("ATT1")]),
            ("KCHILD", [item("ITEM1"), item("ITEM2"), attachment("ATT2", parent="ITEM2")]),
        ]
    )
    assert membership == {"ITEM1": ["KCHILD", "KPARENT"], "ITEM2": ["KCHILD"]}


def test_membership_ignores_attachments_and_notes() -> None:
    """They inherit their parent's folders; listing them would only add noise."""
    note = item("NOTE1", item_type="note", parentItem="ITEM1", note="<p>x</p>")
    membership = item_membership([("K", [item("ITEM1"), attachment("ATT1"), note])])
    assert set(membership) == {"ITEM1"}


def test_planned_attachments_and_notes_carry_their_parents_keys() -> None:
    note = item("NOTE1", item_type="note", parentItem="ITEM1", note="<p>Read this</p>")
    collections = [("Parent", [item("ITEM1"), attachment("ATT1"), note])]
    plan = build_plan(collections, membership={"ITEM1": ["K1", "K2"]})
    assert {a.attachment_key: a.collection_keys for a in plan.attachments} == {
        "ATT1": ["K1", "K2"],
        "NOTE1": ["K1", "K2"],
    }


def test_chunk_metadata_holds_the_keys_as_a_list() -> None:
    plan = build_plan([("P", [item("ITEM1"), attachment("ATT1")])], membership={"ITEM1": ["K1"]})
    assert plan.attachments[0].chunk_metadata(1, 0)["collection_keys"] == ["K1"]


def test_chunk_metadata_omits_the_keys_rather_than_writing_an_empty_list() -> None:
    """Chroma rejects an empty list, so writing one would fail the whole batch."""
    plan = build_plan([("P", [item("ITEM1"), attachment("ATT1")])])
    assert "collection_keys" not in plan.attachments[0].chunk_metadata(1, 0)


# --- files Zotero cannot serve ---------------------------------------------


def unsynced(entry: dict[str, Any]) -> dict[str, Any]:
    """What Zotero sends for a file that was never uploaded: `md5` present and null."""
    entry["data"].update(md5=None, mtime=None)
    return entry


def synced(entry: dict[str, Any]) -> dict[str, Any]:
    entry["data"].update(md5="3def8598b7cced1cecc18e243292d829", mtime=1730672731201)
    return entry


def test_an_unuploaded_file_is_skipped_instead_of_downloaded() -> None:
    plan = build_plan([("C", [item("ITEM1", "Graph RAG"), unsynced(attachment("ATT1"))])])
    assert plan.attachments == []
    assert [entry.reason for entry in plan.skipped] == [SkipReason.NOT_IN_STORAGE]
    # Named by the paper, which is what the user can find in Zotero.
    assert plan.skipped[0].title == "Graph RAG — Full Text PDF"


def test_an_uploaded_file_is_planned() -> None:
    plan = build_plan([("C", [item("ITEM1"), synced(attachment("ATT1"))])])
    assert [a.attachment_key for a in plan.attachments] == ["ATT1"]


def test_a_payload_without_md5_is_still_tried() -> None:
    """No `md5` key says nothing about storage, unlike `md5: null`."""
    plan = build_plan([("C", [item("ITEM1"), attachment("ATT1")])])
    assert [a.attachment_key for a in plan.attachments] == ["ATT1"]


def test_an_unsynced_pdf_gives_way_to_a_synced_snapshot() -> None:
    """The preference picks among files that exist, not files that would 404."""
    collections = [
        (
            "C",
            [
                item("ITEM1"),
                unsynced(attachment("PDF1")),
                synced(attachment("HTML1", content_type="text/html", title="Snapshot")),
            ],
        )
    ]
    plan = build_plan(collections)
    assert [(a.attachment_key, a.document_format) for a in plan.attachments] == [
        ("HTML1", DocumentFormat.HTML)
    ]
    assert {entry.key: entry.reason for entry in plan.skipped} == {
        "PDF1": SkipReason.NOT_IN_STORAGE
    }


def test_an_item_with_nothing_uploaded_is_not_also_called_unreadable() -> None:
    """'No readable attachment' would send the user looking for the wrong problem."""
    collections = [
        (
            "C",
            [
                item("ITEM1"),
                unsynced(attachment("PDF1")),
                unsynced(attachment("HTML1", content_type="text/html")),
            ],
        )
    ]
    plan = build_plan(collections)
    assert sorted(entry.reason for entry in plan.skipped) == [
        SkipReason.NOT_IN_STORAGE,
        SkipReason.NOT_IN_STORAGE,
    ]


def test_a_linked_file_is_skipped_with_its_own_reason() -> None:
    plan = build_plan([("C", [item("ITEM1"), attachment("ATT1", link_mode="linked_file")])])
    assert [entry.reason for entry in plan.skipped] == [SkipReason.LINKED_FILE]


def test_notes_are_unaffected_by_an_unsynced_file() -> None:
    note = item("NOTE1", item_type="note", parentItem="ITEM1", note="<p>My notes</p>")
    plan = build_plan([("C", [item("ITEM1"), unsynced(attachment("ATT1")), note])])
    assert [a.attachment_key for a in plan.attachments] == ["NOTE1"]
