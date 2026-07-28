"""Turn raw Zotero payloads into a plan of what would be indexed.

This is the shared front half of `index`: `--dry-run` prints the plan and stops,
while a real run feeds it to the download/extract/embed pass. Keeping it a pure
function over decoded JSON means the selection rules in PRD §7.1 are testable
without touching the network.
"""

from __future__ import annotations

import logging
from collections import Counter
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from zotero_rag.metadata import authors_str, parse_year, zotero_url

logger = logging.getLogger(__name__)

PDF_CONTENT_TYPE = "application/pdf"


class SkipReason(StrEnum):
    """Why an item or attachment will not be indexed."""

    NO_PDF = "no PDF attachment"
    LINKED_URL = "linked URL (no file)"
    UNSUPPORTED_TYPE = "unsupported content type"
    NOTE = "note"
    DUPLICATE = "already planned (in another collection)"
    ORPHAN_ATTACHMENT = "attachment with no parent item"
    PARENT_OUT_OF_SCOPE = "parent item not in the indexed collections"


class PlannedAttachment(BaseModel):
    """A PDF attachment that a real run would download, extract, and embed."""

    attachment_key: str
    item_key: str
    title: str
    authors: str
    year: str
    doi: str = ""
    collection_name: str
    zotero_url: str = ""
    # Zotero's own version counter for the attachment, bumped whenever the file or
    # its metadata changes. Compared against the stored copy to decide whether an
    # already-indexed attachment needs re-embedding (PRD §11.1).
    version: int = 0

    def chunk_metadata(
        self, page: int, chunk_index: int, embedding_model: str = ""
    ) -> dict[str, str | int]:
        """Per-chunk metadata for the vector store (PRD §7.2).

        Chroma rejects `None`, so every value is a string or an int.

        `embedding_model` and `attachment_version` are what make repair and
        version-aware reindexing possible: the first identifies vectors written by
        a superseded model, the second identifies content that changed in Zotero.
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
            "page": page,
            "chunk_index": chunk_index,
            "embedding_model": embedding_model,
            "attachment_version": self.version,
        }


class SkippedEntry(BaseModel):
    """Something deliberately left out, kept for the run summary."""

    key: str
    title: str
    reason: SkipReason
    collection_name: str = ""


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


def build_plan(
    collections: list[tuple[str, list[dict[str, Any]]]], username: str = ""
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

    parents: dict[str, tuple[dict[str, Any], str]] = {}
    attachments: list[tuple[dict[str, Any], str]] = []

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
                plan.skipped.append(
                    SkippedEntry(
                        key=key,
                        title=_note_title(data),
                        reason=SkipReason.NOTE,
                        collection_name=collection_name,
                    )
                )
            else:
                plan.items_seen += 1
                parents.setdefault(key, (data, collection_name))

    planned_parents: set[str] = set()
    seen_attachments: set[str] = set()

    for data, collection_name in attachments:
        key = data["key"]
        title = data.get("title") or key

        if key in seen_attachments:
            plan.skipped.append(
                SkippedEntry(
                    key=key,
                    title=title,
                    reason=SkipReason.DUPLICATE,
                    collection_name=collection_name,
                )
            )
            continue
        seen_attachments.add(key)

        parent_key = data.get("parentItem")
        if not parent_key:
            logger.warning("Attachment %s has no parent item; skipping", key)
            plan.skipped.append(
                SkippedEntry(
                    key=key,
                    title=title,
                    reason=SkipReason.ORPHAN_ATTACHMENT,
                    collection_name=collection_name,
                )
            )
            continue

        if data.get("linkMode") == "linked_url":
            plan.skipped.append(
                SkippedEntry(
                    key=key,
                    title=title,
                    reason=SkipReason.LINKED_URL,
                    collection_name=collection_name,
                )
            )
            continue

        if data.get("contentType") != PDF_CONTENT_TYPE:
            plan.skipped.append(
                SkippedEntry(
                    key=key,
                    title=title,
                    reason=SkipReason.UNSUPPORTED_TYPE,
                    collection_name=collection_name,
                )
            )
            continue

        parent = parents.get(parent_key)
        if parent is None:
            logger.warning(
                "Attachment %s belongs to item %s, which is outside the indexed "
                "collections; skipping",
                key,
                parent_key,
            )
            plan.skipped.append(
                SkippedEntry(
                    key=key,
                    title=title,
                    reason=SkipReason.PARENT_OUT_OF_SCOPE,
                    collection_name=collection_name,
                )
            )
            continue

        parent_data, parent_collection = parent
        plan.attachments.append(
            PlannedAttachment(
                attachment_key=key,
                item_key=parent_key,
                title=parent_data.get("title") or "(untitled)",
                authors=authors_str(parent_data),
                year=parse_year(parent_data.get("date")),
                doi=parent_data.get("DOI") or "",
                collection_name=parent_collection,
                zotero_url=zotero_url(username, parent_key, parent_data.get("collections")),
                version=_version(data),
            )
        )
        planned_parents.add(parent_key)

    # PRD §7.1: an item with no usable PDF is not indexed, and says so.
    for parent_key, (parent_data, collection_name) in parents.items():
        if parent_key in planned_parents:
            continue
        title = parent_data.get("title") or "(untitled)"
        logger.warning("No PDF attachment for %r (%s); not indexed", title, parent_key)
        plan.skipped.append(
            SkippedEntry(
                key=parent_key,
                title=title,
                reason=SkipReason.NO_PDF,
                collection_name=collection_name,
            )
        )

    return plan


def _version(data: dict[str, Any]) -> int:
    """The attachment's Zotero version, or 0 when the payload omits it.

    0 means "unknown", which makes the comparison in `indexer` treat the
    attachment as unchanged rather than re-embedding it on every run.
    """
    try:
        return int(str(data.get("version")))
    except (TypeError, ValueError):
        return 0


def _note_title(data: dict[str, Any]) -> str:
    """A short label for a note, which has no title field of its own."""
    note = (data.get("note") or "").strip()
    first_line = note.splitlines()[0] if note else ""
    return first_line[:60] or "(note)"
