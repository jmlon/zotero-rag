"""Turn raw Zotero payloads into a plan of what would be indexed.

This is the shared front half of `index`: `--dry-run` prints the plan and stops,
while a real run feeds it to the download/extract/embed pass. Keeping it a pure
function over decoded JSON means the selection rules in PRD §7.1 are testable
without touching the network.

Since M5 an item may offer several readable attachments — a PDF *and* an HTML
snapshot is common — so selection also decides which one to index (§11.2).
"""

from __future__ import annotations

import logging
from collections import Counter
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from zotero_rag import html_text
from zotero_rag.documents import DocumentFormat, LocatorKind, format_for
from zotero_rag.metadata import authors_str, parse_year, zotero_url

logger = logging.getLogger(__name__)

PDF_CONTENT_TYPE = "application/pdf"
LINKED_URL_MODE = "linked_url"

# Order used when config gives none. PDFs first because a published PDF is the
# authoritative copy; a snapshot of the same article is usually the abstract page
# plus site furniture.
DEFAULT_PREFERENCE: tuple[DocumentFormat, ...] = (
    DocumentFormat.PDF,
    DocumentFormat.DOCX,
    DocumentFormat.ODT,
    DocumentFormat.EPUB,
    DocumentFormat.HTML,
    DocumentFormat.TEXT,
)


class SkipReason(StrEnum):
    """Why an item or attachment will not be indexed."""

    NO_CONTENT = "no readable attachment"
    LINKED_URL = "linked URL (no file)"
    UNSUPPORTED_TYPE = "unsupported content type"
    NOT_PREFERRED = "another attachment of this item is preferred"
    NOTE = "note (index_notes is off)"
    DUPLICATE = "already planned (in another collection)"
    ORPHAN_ATTACHMENT = "attachment with no parent item"
    PARENT_OUT_OF_SCOPE = "parent item not in the indexed collections"


class PlannedAttachment(BaseModel):
    """Something a real run would extract, chunk, and embed."""

    attachment_key: str
    item_key: str
    title: str
    authors: str
    year: str
    doi: str = ""
    collection_name: str
    zotero_url: str = ""
    # Zotero's own version counter, bumped whenever the file or its metadata
    # changes. Compared against the stored copy to decide whether an
    # already-indexed attachment needs re-embedding (PRD §11.1).
    version: int = 0
    content_type: str = PDF_CONTENT_TYPE
    document_format: DocumentFormat = DocumentFormat.PDF
    # Notes carry their body in the item payload, so they need no download.
    note_html: str = ""

    @property
    def needs_download(self) -> bool:
        return self.document_format is not DocumentFormat.NOTE

    def chunk_metadata(
        self,
        locator: int,
        chunk_index: int,
        embedding_model: str = "",
        locator_kind: LocatorKind = LocatorKind.PAGE,
        locator_label: str = "",
    ) -> dict[str, str | int]:
        """Per-chunk metadata for the vector store (PRD §7.2).

        Chroma rejects `None`, so every value is a string or an int.

        `page` keeps its name for the locator ordinal: chunks written before M5 are
        all pages, and renaming the key would strand them. `locator_kind` says what
        the ordinal counts, and `locator_label` carries a heading when there is one.
        """
        return {
            "item_key": self.item_key,
            "attachment_key": self.attachment_key,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "zotero_url": self.zotero_url,
            "collection_name": self.collection_name,
            "page": locator,
            "chunk_index": chunk_index,
            "locator_kind": str(locator_kind),
            "locator_label": locator_label,
            "document_format": str(self.document_format),
            "embedding_model": embedding_model,
            "attachment_version": self.version,
        }


class SkippedEntry(BaseModel):
    """Something deliberately left out, kept for the run summary."""

    key: str
    title: str
    reason: SkipReason
    collection_name: str = ""
    # What Zotero said the file was, so "unsupported content type" is actionable.
    content_type: str = ""


class IndexPlan(BaseModel):
    """What a run would do, before any file is fetched."""

    collections: list[str] = Field(default_factory=list)
    attachments: list[PlannedAttachment] = Field(default_factory=list)
    skipped: list[SkippedEntry] = Field(default_factory=list)
    items_seen: int = 0

    @property
    def skip_counts(self) -> dict[SkipReason, int]:
        """Skip totals per reason, ordered most frequent first."""
        counter = Counter(entry.reason for entry in self.skipped)
        return dict(counter.most_common())

    @property
    def item_count(self) -> int:
        """Distinct parent items contributing at least one planned attachment."""
        return len({attachment.item_key for attachment in self.attachments})

    @property
    def format_counts(self) -> dict[DocumentFormat, int]:
        """Planned attachments per format, most frequent first."""
        counter = Counter(attachment.document_format for attachment in self.attachments)
        return dict(counter.most_common())

    def content_types_for(self, reason: SkipReason) -> dict[str, int]:
        """Content types behind one skip reason, so the count is actionable."""
        counter = Counter(
            entry.content_type or "(none)" for entry in self.skipped if entry.reason is reason
        )
        return dict(counter.most_common())

    def skipped_for(self, reason: SkipReason) -> list[SkippedEntry]:
        return [entry for entry in self.skipped if entry.reason is reason]


class _Candidate(BaseModel):
    """A readable attachment competing to represent its parent item."""

    data: dict[str, Any]
    collection_name: str
    document_format: DocumentFormat
    content_type: str


def resolve_preference(names: list[str] | None) -> tuple[DocumentFormat, ...]:
    """Turn configured format names into an order, ignoring unknown entries.

    Formats absent from the list are still indexable — they simply rank last —
    because dropping a readable attachment because of a config omission would be
    a surprising way to lose content.
    """
    if not names:
        return DEFAULT_PREFERENCE
    order: list[DocumentFormat] = []
    for name in names:
        try:
            candidate = DocumentFormat(str(name).strip().lower())
        except ValueError:
            logger.warning("Ignoring unknown format %r in attachment_preference", name)
            continue
        if candidate is not DocumentFormat.NOTE and candidate not in order:
            order.append(candidate)
    for fallback in DEFAULT_PREFERENCE:
        if fallback not in order:
            order.append(fallback)
    return tuple(order)


def build_plan(
    collections: list[tuple[str, list[dict[str, Any]]]],
    username: str = "",
    preference: list[str] | None = None,
    index_all_attachments: bool = False,
    index_notes: bool = True,
) -> IndexPlan:
    """Classify every item in the given collections into planned vs skipped.

    `collections` pairs a collection name with its raw item payloads. Parents are
    collected across *all* collections before attachments are resolved, so an
    attachment whose parent lives in a sibling collection still resolves — the
    reference implementation processed one collection at a time and reported
    spurious cache misses because of it.

    `username` is only needed to build citation URLs; an empty one yields empty
    `zotero_url` values, which keeps `--dry-run` free of an extra API call.
    """
    plan = IndexPlan(collections=[name for name, _ in collections])
    order = resolve_preference(preference)

    parents: dict[str, tuple[dict[str, Any], str]] = {}
    attachments: list[tuple[dict[str, Any], str]] = []
    notes: list[tuple[dict[str, Any], str]] = []

    for collection_name, items in collections:
        for entry in items:
            data = entry.get("data") or {}
            key = data.get("key") or entry.get("key")
            if not key:
                logger.warning("Skipping entry with no key in %r", collection_name)
                continue

            item_type = data.get("itemType")
            if item_type == "attachment":
                attachments.append((data, collection_name))
            elif item_type == "note":
                notes.append((data, collection_name))
            else:
                plan.items_seen += 1
                parents.setdefault(key, (data, collection_name))

    candidates = _collect_candidates(plan, attachments, parents)
    _plan_candidates(plan, candidates, parents, order, username, index_all_attachments)
    _plan_notes(plan, notes, parents, username, index_notes)

    planned_parents = {attachment.item_key for attachment in plan.attachments}
    for parent_key, (parent_data, collection_name) in parents.items():
        if parent_key in planned_parents:
            continue
        title = parent_data.get("title") or "(untitled)"
        logger.warning("No readable attachment for %r (%s); not indexed", title, parent_key)
        plan.skipped.append(
            SkippedEntry(
                key=parent_key,
                title=title,
                reason=SkipReason.NO_CONTENT,
                collection_name=collection_name,
            )
        )

    return plan


def _collect_candidates(
    plan: IndexPlan,
    attachments: list[tuple[dict[str, Any], str]],
    parents: dict[str, tuple[dict[str, Any], str]],
) -> dict[str, list[_Candidate]]:
    """Group readable attachments by parent, recording every rejection."""
    candidates: dict[str, list[_Candidate]] = {}
    seen: set[str] = set()

    for data, collection_name in attachments:
        key = data["key"]
        content_type = str(data.get("contentType") or "")
        entry = SkippedEntry(
            key=key,
            title=data.get("title") or key,
            reason=SkipReason.DUPLICATE,
            collection_name=collection_name,
            content_type=content_type,
        )

        if key in seen:
            plan.skipped.append(entry)
            continue
        seen.add(key)

        if data.get("linkMode") == LINKED_URL_MODE:
            plan.skipped.append(entry.model_copy(update={"reason": SkipReason.LINKED_URL}))
            continue

        document_format = format_for(content_type)
        if document_format is None:
            plan.skipped.append(entry.model_copy(update={"reason": SkipReason.UNSUPPORTED_TYPE}))
            continue

        parent_key = data.get("parentItem")
        if not parent_key:
            logger.warning("Attachment %s has no parent item; skipping", key)
            plan.skipped.append(entry.model_copy(update={"reason": SkipReason.ORPHAN_ATTACHMENT}))
            continue
        if parent_key not in parents:
            logger.warning(
                "Attachment %s belongs to item %s, which is outside the indexed "
                "collections; skipping",
                key,
                parent_key,
            )
            plan.skipped.append(entry.model_copy(update={"reason": SkipReason.PARENT_OUT_OF_SCOPE}))
            continue

        candidates.setdefault(parent_key, []).append(
            _Candidate(
                data=data,
                collection_name=collection_name,
                document_format=document_format,
                content_type=content_type,
            )
        )

    return candidates


def _plan_candidates(
    plan: IndexPlan,
    candidates: dict[str, list[_Candidate]],
    parents: dict[str, tuple[dict[str, Any], str]],
    order: tuple[DocumentFormat, ...],
    username: str,
    index_all_attachments: bool,
) -> None:
    """Pick which of an item's readable attachments to index.

    Roughly a hundred items in the target library hold both a PDF and an HTML
    snapshot of the same article (§11.2). Indexing both would embed the same
    content twice, so by default only the most preferred format wins and the rest
    are recorded as skipped rather than silently dropped.
    """
    for parent_key, group in candidates.items():
        parent_data, _ = parents[parent_key]
        if index_all_attachments:
            chosen, rejected = group, []
        else:
            # The choice is between *formats*, not between files: an item with a
            # paper and its supplement, both PDFs, still gets both indexed (§7.1).
            best_format = min(
                (item.document_format for item in group),
                key=lambda fmt: _rank(fmt, order),
            )
            chosen = [item for item in group if item.document_format is best_format]
            rejected = [item for item in group if item.document_format is not best_format]

        for candidate in chosen:
            plan.attachments.append(_planned(candidate, parent_key, parent_data, username))
        for candidate in rejected:
            plan.skipped.append(
                SkippedEntry(
                    key=candidate.data["key"],
                    title=candidate.data.get("title") or candidate.data["key"],
                    reason=SkipReason.NOT_PREFERRED,
                    collection_name=candidate.collection_name,
                    content_type=candidate.content_type,
                )
            )


def _rank(document_format: DocumentFormat, order: tuple[DocumentFormat, ...]) -> int:
    return order.index(document_format) if document_format in order else len(order)


def _planned(
    candidate: _Candidate,
    parent_key: str,
    parent_data: dict[str, Any],
    username: str,
) -> PlannedAttachment:
    return PlannedAttachment(
        attachment_key=candidate.data["key"],
        item_key=parent_key,
        title=parent_data.get("title") or "(untitled)",
        authors=authors_str(parent_data),
        year=parse_year(parent_data.get("date")),
        doi=parent_data.get("DOI") or "",
        collection_name=candidate.collection_name,
        zotero_url=zotero_url(username, parent_key, parent_data.get("collections")),
        version=_version(candidate.data),
        content_type=candidate.content_type,
        document_format=candidate.document_format,
    )


def _plan_notes(
    plan: IndexPlan,
    notes: list[tuple[dict[str, Any], str]],
    parents: dict[str, tuple[dict[str, Any], str]],
    username: str,
    index_notes: bool,
) -> None:
    """Index Zotero notes alongside their item's file, not instead of it.

    A note is a reader's own commentary, so it complements the paper rather than
    competing with it, and it takes no part in the format preference. Notes come
    with their body in the payload, so they cost no download.
    """
    seen: set[str] = set()
    for data, collection_name in notes:
        key = data["key"]
        title = _note_title(data)
        if key in seen:
            continue
        seen.add(key)

        parent_key = data.get("parentItem")
        body = data.get("note") or ""
        if not index_notes or not body.strip() or parent_key not in parents:
            # A standalone note has no bibliographic metadata to cite, so it stays
            # out until notes become first-class (a later milestone).
            plan.skipped.append(
                SkippedEntry(
                    key=key,
                    title=title,
                    reason=SkipReason.NOTE,
                    collection_name=collection_name,
                    content_type="note",
                )
            )
            continue

        parent_data, _ = parents[str(parent_key)]
        plan.attachments.append(
            PlannedAttachment(
                attachment_key=key,
                item_key=str(parent_key),
                title=parent_data.get("title") or "(untitled)",
                authors=authors_str(parent_data),
                year=parse_year(parent_data.get("date")),
                doi=parent_data.get("DOI") or "",
                collection_name=collection_name,
                zotero_url=zotero_url(username, str(parent_key), parent_data.get("collections")),
                version=_version(data),
                content_type="note",
                document_format=DocumentFormat.NOTE,
                note_html=body,
            )
        )


def _version(data: dict[str, Any]) -> int:
    """The item's Zotero version, or 0 when the payload omits it.

    0 means "unknown", which makes the comparison in `indexer` treat the
    attachment as unchanged rather than re-embedding it on every run.
    """
    try:
        return int(str(data.get("version")))
    except (TypeError, ValueError):
        return 0


def _note_title(data: dict[str, Any]) -> str:
    """A short label for a note, which has no title field of its own.

    Note bodies are HTML, so the markup is stripped first: `<p>Read this</p>`
    should show as `Read this`, not as a tag.
    """
    text = html_text.to_text(data.get("note") or "").strip()
    first_line = text.splitlines()[0] if text else ""
    return first_line[:60] or "(note)"
