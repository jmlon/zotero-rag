"""Selection rules from PRD §7.1, applied to raw Zotero item payloads."""

from __future__ import annotations

from typing import Any

from zotero_rag.planning import SkipReason, build_plan


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
    assert SkipReason.NO_PDF in reasons


def test_non_pdf_attachment_is_skipped() -> None:
    plan = build_plan([("C", [item("ITEM1"), attachment("ATT1", content_type="text/html")])])
    assert plan.attachments == []
    assert SkipReason.UNSUPPORTED_TYPE in {entry.reason for entry in plan.skipped}


def test_epub_is_skipped_in_the_poc() -> None:
    plan = build_plan(
        [("C", [item("ITEM1"), attachment("ATT1", content_type="application/epub+zip")])]
    )
    assert plan.attachments == []
    assert SkipReason.UNSUPPORTED_TYPE in {entry.reason for entry in plan.skipped}


def test_item_without_any_attachment_is_reported() -> None:
    plan = build_plan([("C", [item("ITEM1", "Lonely Item")])])
    assert plan.attachments == []
    assert len(plan.skipped) == 1
    assert plan.skipped[0].reason == SkipReason.NO_PDF
    assert plan.skipped[0].title == "Lonely Item"


def test_notes_are_skipped() -> None:
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
    assert len(plan.attachments) == 1
    skipped = [entry for entry in plan.skipped if entry.reason == SkipReason.NOTE]
    assert len(skipped) == 1
    assert skipped[0].title == "Remember to cite this"
    # A note must not make its parent look item-less.
    assert plan.items_seen == 1


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
    assert len(plan.attachments) == 1
    counts = plan.skip_counts
    assert counts[SkipReason.NO_PDF] == 2
    assert counts[SkipReason.UNSUPPORTED_TYPE] == 1


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
