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
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from zotero_rag_core import html_text, provenance
from zotero_rag_core.documents import DocumentFormat, LocatorKind, format_for
from zotero_rag_core.metadata import (
    authors_str,
    citation_url,
    item_tags,
    parse_year,
    year_number,
)

logger = logging.getLogger(__name__)

PDF_CONTENT_TYPE = "application/pdf"
LINKED_URL_MODE = "linked_url"
LINKED_FILE_MODE = "linked_file"

# Order used when config gives none. Markdown first: a transcript made by
# `convert` is the same document as its source, and indexing it measured equal
# in answer quality for 13% fewer prompt tokens (PLAN-markdown, Phase 6). It only
# chooses between the two copies of a converted document — an item without a
# transcript still ranks its own files below. Then PDFs, because a published PDF
# is the authoritative copy; a snapshot is usually the abstract page plus site
# furniture.
DEFAULT_PREFERENCE: tuple[DocumentFormat, ...] = (
    DocumentFormat.MARKDOWN,
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
    LINKED_FILE = "linked file (kept only on your computer)"
    NOT_IN_STORAGE = "file not uploaded to Zotero storage"
    UNSUPPORTED_TYPE = "unsupported content type"
    NOT_PREFERRED = "another attachment of this item is preferred"
    NOTE = "note (index_notes is off)"
    DUPLICATE = "already planned (in another collection)"
    ORPHAN_ATTACHMENT = "attachment with no parent item"
    PARENT_OUT_OF_SCOPE = "parent item not in the indexed collections"
    # `convert` only: plain text, Markdown and CSV need no transcript.
    ALREADY_TEXT = "already plain text"
    # A transcript and its source hold the same text; only one of them is indexed.
    SAME_DOCUMENT = "the same document is indexed from its transcript or source"


class PlannedAttachment(BaseModel):
    """Something a real run would extract, chunk, and embed."""

    attachment_key: str
    item_key: str
    #: Zotero's item type of the parent: `journalArticle`, `book`, `thesis`, ...
    item_type: str = ""
    title: str
    authors: str
    year: str
    doi: str = ""
    collection_name: str
    # Every indexed collection holding the parent item, so a search can be limited
    # to a subtree (PRD-CursesUI §4.3). Empty only for plans built without
    # membership, such as in tests; such chunks are simply left out of a scoped search.
    collection_keys: list[str] = Field(default_factory=list)
    #: The parent item's tags, sorted, for filtering by tag.
    tags: list[str] = Field(default_factory=list)
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
    ) -> dict[str, str | int | list[str]]:
        """Per-chunk metadata for the vector store (PRD §7.2).

        Chroma rejects `None`, so every value is a string or an int, except
        `collection_keys` and `tags`, lists of strings. Chroma also rejects an empty
        list, so those keys are left out rather than written empty.

        `year_int` is the year as a number, for range filters (`store.Filters`), and
        `store.YEAR_UNKNOWN` when the item has none. `year` stays the string cited.

        `page` keeps its name for the locator ordinal: chunks written before M5 are
        all pages, and renaming the key would strand them. `locator_kind` says what
        the ordinal counts, and `locator_label` carries a heading when there is one.
        """
        metadata: dict[str, str | int | list[str]] = {
            "item_key": self.item_key,
            "item_type": self.item_type,
            "attachment_key": self.attachment_key,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "year_int": year_number(self.year),
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
        if self.collection_keys:
            metadata["collection_keys"] = list(self.collection_keys)
        if self.tags:
            metadata["tags"] = list(self.tags)
        return metadata


class SkippedEntry(BaseModel):
    """Something deliberately left out, kept for the run summary."""

    key: str
    title: str
    reason: SkipReason
    collection_name: str = ""
    # What Zotero said the file was, so "unsupported content type" is actionable.
    content_type: str = ""
    # SAME_DOCUMENT: the attachment indexed in this one's place.
    replaced_by: str = ""


class SkipReport(BaseModel):
    """What a plan deliberately left out, summarised for the run report.

    Shared by `index` and `convert` plans, so one printer serves both.
    """

    collections: list[str] = Field(default_factory=list)
    skipped: list[SkippedEntry] = Field(default_factory=list)
    items_seen: int = 0

    @property
    def skip_counts(self) -> dict[SkipReason, int]:
        """Skip totals per reason, ordered most frequent first."""
        counter = Counter(entry.reason for entry in self.skipped)
        return dict(counter.most_common())

    def content_types_for(self, reason: SkipReason) -> dict[str, int]:
        """Content types behind one skip reason, so the count is actionable."""
        counter = Counter(
            entry.content_type or "(none)" for entry in self.skipped if entry.reason is reason
        )
        return dict(counter.most_common())

    def skipped_for(self, reason: SkipReason) -> list[SkippedEntry]:
        return [entry for entry in self.skipped if entry.reason is reason]


class IndexPlan(SkipReport):
    """What a run would do, before any file is fetched."""

    attachments: list[PlannedAttachment] = Field(default_factory=list)

    @property
    def item_count(self) -> int:
        """Distinct parent items contributing at least one planned attachment."""
        return len({attachment.item_key for attachment in self.attachments})

    @property
    def format_counts(self) -> dict[DocumentFormat, int]:
        """Planned attachments per format, most frequent first."""
        counter = Counter(attachment.document_format for attachment in self.attachments)
        return dict(counter.most_common())


class Candidate(BaseModel):
    """A readable attachment competing to represent its parent item.

    `format` is whatever the caller's format function returned: a `DocumentFormat`
    value for indexing, or also a conversion-only format for `convert`.
    """

    data: dict[str, Any]
    collection_name: str
    format: str
    content_type: str


class Candidates(BaseModel):
    """Attachments grouped by parent, with every rejection recorded."""

    by_parent: dict[str, list[Candidate]] = Field(default_factory=dict)
    # Parents with a readable attachment Zotero cannot serve (linked or unsynced).
    unavailable: set[str] = Field(default_factory=set)
    skipped: list[SkippedEntry] = Field(default_factory=list)


FormatFn = Callable[[str | None], str | None]


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


def item_membership(
    collections: Iterable[tuple[str, list[dict[str, Any]]]],
) -> dict[str, list[str]]:
    """Map each parent item key to the keys of the fetched collections holding it.

    `collections` pairs a collection *key* with its raw item payloads. Fetching a
    collection returns every item in it, so this is exact within the indexed scope.
    The item's own `collections` field would also list collections outside the scope,
    which a search could never select. Attachments and notes are left out because
    they inherit their parent's membership.
    """
    membership: dict[str, set[str]] = {}
    for collection_key, items in collections:
        for entry in items:
            data = entry.get("data") or {}
            key = data.get("key") or entry.get("key")
            if key and data.get("itemType") not in ("attachment", "note"):
                membership.setdefault(str(key), set()).add(collection_key)
    return {item_key: sorted(keys) for item_key, keys in membership.items()}


def library_membership(items: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each parent item key to the collections it is filed in, library-wide.

    When the whole library is indexed every collection is in scope, so an item's
    own `collections` field is exact, and fetching each collection separately
    (`item_membership`) would cost one request per collection for the same answer.
    Unfiled items map to an empty list.
    """
    membership: dict[str, list[str]] = {}
    for entry in items:
        data = entry.get("data") or {}
        key = data.get("key") or entry.get("key")
        if key and data.get("itemType") not in ("attachment", "note"):
            membership[str(key)] = sorted({str(k) for k in data.get("collections") or []})
    return membership


def build_plan(
    collections: list[tuple[str, list[dict[str, Any]]]],
    username: str = "",
    preference: list[str] | None = None,
    index_all_attachments: bool = False,
    index_notes: bool = True,
    membership: Mapping[str, list[str]] | None = None,
) -> IndexPlan:
    """Classify every item in the given collections into planned vs skipped.

    `collections` pairs a collection name with its raw item payloads. Parents are
    collected across *all* collections before attachments are resolved, so an
    attachment whose parent lives in a sibling collection still resolves — the
    reference implementation processed one collection at a time and reported
    spurious cache misses because of it.

    `username` builds citation URLs inside the item's first collection
    (`metadata.zotero_url`), which suits a user library. Without one, the URL is
    the item's own web link from the API payload (`links.alternate`), which is
    also right for a group library. A payload without links yields an empty URL.

    `membership`, from `item_membership`, gives each planned attachment the
    `collection_keys` of its parent item.
    """
    plan = IndexPlan(collections=[name for name, _ in collections])
    order = resolve_preference(preference)
    membership = membership or {}

    parents: dict[str, tuple[dict[str, Any], str]] = {}
    urls: dict[str, str] = {}
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
                if key not in parents:
                    parents[key] = (data, collection_name)
                    urls[key] = citation_url(username, entry)

    collected = collect_candidates(attachments, parents, _index_format)
    plan.skipped.extend(collected.skipped)
    unavailable = collected.unavailable
    _plan_candidates(
        plan, collected.by_parent, parents, order, urls, index_all_attachments, membership
    )
    _plan_notes(plan, notes, parents, urls, index_notes, membership)

    planned_parents = {attachment.item_key for attachment in plan.attachments}
    for parent_key, (parent_data, collection_name) in parents.items():
        # An item whose files exist but cannot be downloaded is already reported,
        # with its fix, under the attachment's own reason. Calling it "no readable
        # attachment" as well would send the user looking for the wrong problem.
        if parent_key in planned_parents or parent_key in unavailable:
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


def is_in_storage(data: dict[str, Any]) -> bool:
    """Whether Zotero holds this attachment's file, so the Web API can serve it.

    Zotero sends `md5: null` (and `mtime: null`) for a file that was never uploaded:
    file sync off, the storage quota full, files kept on WebDAV, or a sync that has
    not finished yet. Downloading such a file only gets a 404. A payload with no
    `md5` key at all says nothing either way, so it is still tried.
    """
    return "md5" not in data or bool(data["md5"])


def _index_format(content_type: str | None) -> str | None:
    document_format = format_for(content_type)
    return None if document_format is None else document_format.value


def collect_candidates(
    attachments: list[tuple[dict[str, Any], str]],
    parents: dict[str, tuple[dict[str, Any], str]],
    format_of: FormatFn,
) -> Candidates:
    """Group downloadable, readable attachments by parent, recording every rejection.

    `format_of` decides what "readable" means: `index` and `convert` read different
    sets of formats. Also returns the parents that had a readable attachment
    Zotero cannot serve. Leaving those out *before* the format preference runs
    means that an unsynced PDF gives way to a snapshot that was synced, instead of
    winning and then failing to download.
    """
    result = Candidates()
    candidates = result.by_parent
    unavailable = result.unavailable
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
            result.skipped.append(entry)
            continue
        seen.add(key)

        link_mode = data.get("linkMode")
        if link_mode == LINKED_URL_MODE:
            result.skipped.append(entry.model_copy(update={"reason": SkipReason.LINKED_URL}))
            continue

        document_format = format_of(content_type)
        if document_format is None:
            result.skipped.append(entry.model_copy(update={"reason": SkipReason.UNSUPPORTED_TYPE}))
            continue

        parent_key = data.get("parentItem")
        if not parent_key:
            logger.warning("Attachment %s has no parent item; skipping", key)
            result.skipped.append(entry.model_copy(update={"reason": SkipReason.ORPHAN_ATTACHMENT}))
            continue
        if parent_key not in parents:
            logger.warning(
                "Attachment %s belongs to item %s, which is outside the indexed "
                "collections; skipping",
                key,
                parent_key,
            )
            result.skipped.append(
                entry.model_copy(update={"reason": SkipReason.PARENT_OUT_OF_SCOPE})
            )
            continue

        if link_mode == LINKED_FILE_MODE or not is_in_storage(data):
            reason = (
                SkipReason.LINKED_FILE
                if link_mode == LINKED_FILE_MODE
                else (SkipReason.NOT_IN_STORAGE)
            )
            # "Snapshot" or "Preprint PDF" alone cannot be found in Zotero; the
            # paper's title can, and the user has to go and fix it there.
            paper = parents[parent_key][0].get("title") or "(untitled)"
            result.skipped.append(
                entry.model_copy(update={"reason": reason, "title": f"{paper} — {entry.title}"})
            )
            unavailable.add(parent_key)
            continue

        candidates.setdefault(parent_key, []).append(
            Candidate(
                data=data,
                collection_name=collection_name,
                format=document_format,
                content_type=content_type,
            )
        )

    return result


def _plan_candidates(
    plan: IndexPlan,
    candidates: dict[str, list[Candidate]],
    parents: dict[str, tuple[dict[str, Any], str]],
    order: tuple[DocumentFormat, ...],
    urls: Mapping[str, str],
    index_all_attachments: bool,
    membership: Mapping[str, list[str]],
) -> None:
    """Pick which of an item's readable attachments to index.

    Roughly a hundred items in the target library hold both a PDF and an HTML
    snapshot of the same article (§11.2). Indexing both would embed the same
    content twice, so by default only the most preferred format wins and the rest
    are recorded as skipped rather than silently dropped.

    A Markdown transcript made by `convert` and its source are one document in two
    copies. The document competes under its source's format — so a paper's
    transcript does not push out its supplement — and the preference order then
    picks which copy is indexed. Never both, even with `index_all_attachments`.
    """
    for parent_key, group in candidates.items():
        parent_data, _ = parents[parent_key]
        documents, copies, extra = _pair_transcripts(group)
        chosen, rejected = choose_preferred(documents, order, every=index_all_attachments)

        for document in chosen:
            pair = copies.get(document.data["key"])
            if pair is None:
                indexed, other = document, None
            else:
                source, transcript = pair
                prefer_transcript = _rank(DocumentFormat.MARKDOWN, order) < _rank(
                    source.format, order
                )
                indexed, other = (transcript, source) if prefer_transcript else pair
            plan.attachments.append(
                _planned(indexed, parent_key, parent_data, urls[parent_key], membership)
            )
            if other is not None:
                plan.skipped.append(_skipped(other, SkipReason.SAME_DOCUMENT, indexed))
            for duplicate in extra.get(document.data["key"], []):
                plan.skipped.append(_skipped(duplicate, SkipReason.SAME_DOCUMENT, indexed))
        for document in rejected:
            plan.skipped.append(_skipped(document, SkipReason.NOT_PREFERRED))
            pair = copies.get(document.data["key"])
            for copy in ([pair[1]] if pair else []) + extra.get(document.data["key"], []):
                plan.skipped.append(_skipped(copy, SkipReason.NOT_PREFERRED))


def _pair_transcripts(
    group: list[Candidate],
) -> tuple[list[Candidate], dict[str, tuple[Candidate, Candidate]], dict[str, list[Candidate]]]:
    """Fold each transcript into the document it transcribes.

    Returns the documents that compete for the item — sources, plain attachments,
    and transcripts whose source is gone — then, keyed by the competing document,
    the (source, transcript) pair, and any surplus transcripts of the same source.
    A transcript whose source is gone competes under the source's format, as
    recorded in its provenance, so a converted PDF keeps the PDF's standing.
    """
    by_key = {str(candidate.data["key"]): candidate for candidate in group}
    transcripts: dict[str, list[Candidate]] = {}
    documents: list[Candidate] = []
    for candidate in group:
        record = (
            provenance.parse(candidate.data.get("note"))
            if candidate.format == DocumentFormat.MARKDOWN
            else None
        )
        if record is None:
            documents.append(candidate)
        elif record.source in by_key:
            transcripts.setdefault(record.source, []).append(candidate)
        else:
            standing = record.source_format or candidate.format
            documents.append(candidate.model_copy(update={"format": standing}))

    copies: dict[str, tuple[Candidate, Candidate]] = {}
    extra: dict[str, list[Candidate]] = {}
    for source_key, found in transcripts.items():
        # Two transcripts of one source only follow an interrupted run; the
        # newest stands for the pair and the rest are skipped, never indexed twice.
        found.sort(key=lambda c: _version(c.data), reverse=True)
        copies[source_key] = (by_key[source_key], found[0])
        extra[source_key] = found[1:]
    return documents, copies, extra


def _skipped(
    candidate: Candidate, reason: SkipReason, replaced_by: Candidate | None = None
) -> SkippedEntry:
    return SkippedEntry(
        key=candidate.data["key"],
        title=candidate.data.get("title") or candidate.data["key"],
        reason=reason,
        collection_name=candidate.collection_name,
        content_type=candidate.content_type,
        replaced_by=str(replaced_by.data["key"]) if replaced_by is not None else "",
    )


def choose_preferred(
    group: list[Candidate], order: Sequence[str], *, every: bool = False
) -> tuple[list[Candidate], list[Candidate]]:
    """Split an item's candidates into (chosen, rejected) by format preference.

    The choice is between *formats*, not between files: an item with a paper and
    its supplement, both PDFs, keeps both (§7.1). A format missing from `order`
    ranks last rather than being dropped. `every` keeps them all.
    """
    if every:
        return list(group), []
    best = min((item.format for item in group), key=lambda fmt: _rank(fmt, order))
    return (
        [item for item in group if item.format == best],
        [item for item in group if item.format != best],
    )


def _rank(document_format: str, order: Sequence[str]) -> int:
    return list(order).index(document_format) if document_format in order else len(order)


def _planned(
    candidate: Candidate,
    parent_key: str,
    parent_data: dict[str, Any],
    url: str,
    membership: Mapping[str, list[str]],
) -> PlannedAttachment:
    return PlannedAttachment(
        attachment_key=candidate.data["key"],
        item_key=parent_key,
        item_type=str(parent_data.get("itemType") or ""),
        title=parent_data.get("title") or "(untitled)",
        authors=authors_str(parent_data),
        year=parse_year(parent_data.get("date")),
        doi=parent_data.get("DOI") or "",
        collection_name=candidate.collection_name,
        collection_keys=list(membership.get(parent_key, [])),
        tags=item_tags(parent_data),
        zotero_url=url,
        version=_version(candidate.data),
        content_type=candidate.content_type,
        # The file's own format, not its standing: a transcript whose source is
        # gone competes as that source's format but is read as Markdown.
        document_format=format_for(candidate.content_type) or DocumentFormat(candidate.format),
    )


def _plan_notes(
    plan: IndexPlan,
    notes: list[tuple[dict[str, Any], str]],
    parents: dict[str, tuple[dict[str, Any], str]],
    urls: Mapping[str, str],
    index_notes: bool,
    membership: Mapping[str, list[str]],
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
                item_type=str(parent_data.get("itemType") or ""),
                title=parent_data.get("title") or "(untitled)",
                authors=authors_str(parent_data),
                year=parse_year(parent_data.get("date")),
                doi=parent_data.get("DOI") or "",
                collection_name=collection_name,
                collection_keys=list(membership.get(str(parent_key), [])),
                tags=item_tags(parent_data),
                zotero_url=urls[str(parent_key)],
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
