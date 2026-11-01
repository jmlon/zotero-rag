"""Which attachments need a Markdown transcript, and what to do about each.

This is the front half of `convert` (PLAN-markdown, Phase 3), and like
`planning.build_plan` it is a pure function over the raw item payloads, so every
rule is testable without the network. `--dry-run` prints the plan and stops.

Every source attachment ends in one state. Only three lead to work: MISSING
(create a transcript), STALE (the source was replaced, regenerate), and — only
when asked — OUTDATED (a newer converter) and EDITED (a person changed the
transcript; `--force` overwrites). A transcript is matched to its source by the
provenance note on it, never by file name, and one this tool did not write is
never touched.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from zotero_rag.markdown import converter_name
from zotero_rag_core import provenance
from zotero_rag_core.documents import DocumentFormat, conversion_format_for, format_for
from zotero_rag_core.metadata import authors_str, parse_year, zotero_url
from zotero_rag_core.planning import (
    LINKED_URL_MODE,
    Candidate,
    SkippedEntry,
    SkipReason,
    SkipReport,
    choose_preferred,
    collect_candidates,
    resolve_preference,
)
from zotero_rag_core.provenance import Provenance

MARKDOWN_TYPES = frozenset({"text/markdown", "text/x-markdown"})
MARKDOWN_SUFFIXES = (".md", ".markdown")
STORED_MODES = frozenset({"imported_file", "imported_url"})


class ConversionState(StrEnum):
    CURRENT = "up to date"
    MISSING = "no transcript yet"
    STALE = "source changed since conversion"
    OUTDATED = "made by an older converter"
    EDITED = "transcript edited in Zotero"
    USER_PROVIDED = "Markdown attached by the user"
    # Transcripts whose source is not (or no longer) the one being converted.
    SOURCE_GONE = "source removed by shrink-storage"
    SOURCE_MISSING = "source no longer in Zotero"
    SUPERSEDED = "source is no longer the attachment converted"
    DUPLICATE = "a second transcript of the same source"


class ConversionAction(StrEnum):
    CREATE = "create"
    REGENERATE = "regenerate"
    NONE = "none"


class SourceAttachment(BaseModel):
    """An attachment to convert, with the parent's metadata for the front matter."""

    attachment_key: str
    item_key: str
    title: str
    authors: str
    year: str
    doi: str = ""
    zotero_url: str = ""
    collection_name: str
    content_type: str
    source_format: str
    filename: str = ""
    md5: str = ""
    version: int = 0

    @property
    def document_format(self) -> DocumentFormat | None:
        """The format `index` knows it by, if it reads it at all (for the cache)."""
        return format_for(self.content_type)


class Transcript(BaseModel):
    """A Markdown attachment as Zotero holds it now."""

    key: str
    version: int = 0
    md5: str = ""
    filename: str = ""
    link_mode: str = ""
    # None for Markdown the user attached; theirs, never ours to change.
    provenance: Provenance | None = None


class ConversionEntry(BaseModel):
    state: ConversionState
    action: ConversionAction = ConversionAction.NONE
    source: SourceAttachment | None = None
    transcript: Transcript | None = None
    user_markdown: list[Transcript] = Field(default_factory=list)
    # For transcripts without a source in this plan: the parent paper's title.
    item_title: str = ""

    @property
    def title(self) -> str:
        return self.source.title if self.source else self.item_title


class ConversionPlan(SkipReport):
    entries: list[ConversionEntry] = Field(default_factory=list)

    @property
    def actions(self) -> list[ConversionEntry]:
        """The entries a real run would download, convert and upload."""
        return [entry for entry in self.entries if entry.action is not ConversionAction.NONE]

    @property
    def state_counts(self) -> dict[ConversionState, int]:
        return dict(Counter(entry.state for entry in self.entries).most_common())


def build_conversion_plan(
    collections: list[tuple[str, list[dict[str, Any]]]],
    *,
    username: str = "",
    preference: list[str] | None = None,
    every_attachment: bool = False,
    force: bool = False,
    reconvert: bool = False,
    converter_for: Callable[[str], str] = converter_name,
) -> ConversionPlan:
    """Classify every attachment in the given collections.

    `preference` and `every_attachment` follow the index settings of the same
    meaning, so the transcript is made from the attachment `index` would read.
    `force` regenerates transcripts edited in Zotero; `reconvert` regenerates those
    made by another converter version. `converter_for` names the converter a
    format would use now, which is what OUTDATED is judged against.
    """
    plan = ConversionPlan(collections=[name for name, _ in collections])
    parents, attachments = _split(collections, plan)

    managed: dict[str, list[Transcript]] = {}
    user_markdown: dict[str, list[Transcript]] = {}
    sources: list[tuple[dict[str, Any], str]] = []
    # Every attachment key, and whether it went on to candidate collection.
    routed: dict[str, bool] = {}
    # Papers with a plain-text attachment are readable already, not contentless.
    has_text: set[str] = set()
    for data, collection_name in attachments:
        key = str(data["key"])
        if key in routed:
            # An item filed in two collections lists its attachments twice. A
            # repeated source goes through again so it is reported as a duplicate,
            # exactly as `index` reports it; anything else is counted once.
            if routed[key]:
                sources.append((data, collection_name))
            continue
        parent = str(data.get("parentItem") or "")
        record = provenance.parse(data.get("note")) if _is_stored(data) else None
        routed[key] = False
        if record is not None or _is_markdown(data):
            # A transcript is never a source. Outside the target it is simply
            # not this run's business.
            if parent in parents:
                bucket = managed if record is not None else user_markdown
                bucket.setdefault(parent, []).append(_transcript(data, record))
        elif format_for(data.get("contentType")) is DocumentFormat.TEXT:
            plan.skipped.append(_skip(data, collection_name, SkipReason.ALREADY_TEXT))
            has_text.add(parent)
        else:
            routed[key] = True
            sources.append((data, collection_name))

    collected = collect_candidates(sources, parents, conversion_format_for)
    plan.skipped.extend(collected.skipped)
    order = resolve_preference(preference)

    matched: set[str] = set()
    not_chosen: set[str] = set()
    for parent_key, group in collected.by_parent.items():
        parent_data, _ = parents[parent_key]
        chosen, rejected = choose_preferred(group, order, every=every_attachment)
        for candidate in rejected:
            not_chosen.add(str(candidate.data["key"]))
            plan.skipped.append(
                _skip(candidate.data, candidate.collection_name, SkipReason.NOT_PREFERRED)
            )
        for candidate in chosen:
            source = _source(candidate, parent_key, parent_data, username)
            entry, used = _classify(
                source,
                managed.get(parent_key, []),
                user_markdown.get(parent_key, []),
                force=force,
                reconvert=reconvert,
                converter=converter_for(source.source_format),
            )
            plan.entries.extend(entry)
            matched.update(used)

    _account_for_unmatched(plan, managed, parents, matched, not_chosen, set(routed))

    described = {entry.source.item_key for entry in plan.entries if entry.source}
    described |= set(managed) | set(user_markdown) | has_text
    for parent_key, (parent_data, collection_name) in parents.items():
        if parent_key in described or parent_key in collected.unavailable:
            continue
        plan.skipped.append(
            SkippedEntry(
                key=parent_key,
                title=parent_data.get("title") or "(untitled)",
                reason=SkipReason.NO_CONTENT,
                collection_name=collection_name,
            )
        )
    return plan


def _classify(
    source: SourceAttachment,
    transcripts: list[Transcript],
    user_markdown: list[Transcript],
    *,
    force: bool,
    reconvert: bool,
    converter: str,
) -> tuple[list[ConversionEntry], set[str]]:
    """The entry for one chosen source, plus any duplicate transcripts of it."""
    mine = [t for t in transcripts if t.provenance and t.provenance.source == source.attachment_key]
    used = {t.key for t in mine}

    # D3: Markdown the user attached means the item is converted. Ours, if any,
    # stays as it is — `--force` included — rather than competing with theirs.
    if user_markdown:
        return [
            ConversionEntry(
                state=ConversionState.USER_PROVIDED,
                source=source,
                transcript=mine[0] if mine else None,
                user_markdown=user_markdown,
            )
        ], used

    if not mine:
        entry = ConversionEntry(
            state=ConversionState.MISSING, action=ConversionAction.CREATE, source=source
        )
        return [entry], used

    # Two transcripts of one source should not happen, but a run interrupted
    # between upload and bookkeeping could leave one. The newest is the one kept
    # current; the rest are reported, not deleted.
    mine.sort(key=lambda t: t.version, reverse=True)
    current, extras = mine[0], mine[1:]
    record = current.provenance
    assert record is not None  # `mine` holds only transcripts with provenance

    if current.md5 != record.md5:
        state = ConversionState.EDITED
        action = ConversionAction.REGENERATE if force else ConversionAction.NONE
    elif source.md5 and record.source_md5 != source.md5:
        state, action = ConversionState.STALE, ConversionAction.REGENERATE
    elif record.converter != converter:
        state = ConversionState.OUTDATED
        action = ConversionAction.REGENERATE if reconvert else ConversionAction.NONE
    else:
        state, action = ConversionState.CURRENT, ConversionAction.NONE

    entries = [ConversionEntry(state=state, action=action, source=source, transcript=current)]
    entries.extend(
        ConversionEntry(state=ConversionState.DUPLICATE, transcript=extra, item_title=source.title)
        for extra in extras
    )
    return entries, used


def _account_for_unmatched(
    plan: ConversionPlan,
    managed: dict[str, list[Transcript]],
    parents: dict[str, tuple[dict[str, Any], str]],
    matched: set[str],
    not_chosen: set[str],
    present: set[str],
) -> None:
    """Report transcripts no chosen source claimed. None of them is touched."""
    for parent_key, transcripts in managed.items():
        title = parents[parent_key][0].get("title") or "(untitled)"
        for transcript in transcripts:
            if transcript.key in matched:
                continue
            record = transcript.provenance
            assert record is not None
            if record.source_removed is not None:
                state = ConversionState.SOURCE_GONE
            elif record.source in not_chosen or record.source in present:
                state = ConversionState.SUPERSEDED
            else:
                state = ConversionState.SOURCE_MISSING
            plan.entries.append(
                ConversionEntry(state=state, transcript=transcript, item_title=title)
            )


def _split(
    collections: list[tuple[str, list[dict[str, Any]]]], plan: ConversionPlan
) -> tuple[dict[str, tuple[dict[str, Any], str]], list[tuple[dict[str, Any], str]]]:
    """Parents (first collection wins) and attachments; notes play no part here."""
    parents: dict[str, tuple[dict[str, Any], str]] = {}
    attachments: list[tuple[dict[str, Any], str]] = []
    for collection_name, items in collections:
        for entry in items:
            data = entry.get("data") or {}
            key = data.get("key") or entry.get("key")
            if not key:
                continue
            item_type = data.get("itemType")
            if item_type == "attachment":
                attachments.append(({**data, "key": key}, collection_name))
            elif item_type not in ("note", "annotation"):
                if key not in parents:
                    plan.items_seen += 1
                parents.setdefault(str(key), (data, collection_name))
    return parents, attachments


def _is_stored(data: dict[str, Any]) -> bool:
    return data.get("linkMode") in STORED_MODES


def _is_markdown(data: dict[str, Any]) -> bool:
    """Markdown by content type or by name: Zotero often stores `.md` as text/plain."""
    if data.get("linkMode") == LINKED_URL_MODE:
        return False
    content_type = str(data.get("contentType") or "").split(";", 1)[0].strip().lower()
    if content_type in MARKDOWN_TYPES:
        return True
    name = str(data.get("filename") or data.get("path") or "").lower()
    return name.endswith(MARKDOWN_SUFFIXES)


def _transcript(data: dict[str, Any], record: Provenance | None) -> Transcript:
    return Transcript(
        key=str(data["key"]),
        version=_int(data.get("version")),
        md5=str(data.get("md5") or ""),
        filename=str(data.get("filename") or ""),
        link_mode=str(data.get("linkMode") or ""),
        provenance=record,
    )


def _source(
    candidate: Candidate, parent_key: str, parent_data: dict[str, Any], username: str
) -> SourceAttachment:
    data = candidate.data
    return SourceAttachment(
        attachment_key=str(data["key"]),
        item_key=parent_key,
        title=parent_data.get("title") or "(untitled)",
        authors=authors_str(parent_data),
        year=parse_year(parent_data.get("date")),
        doi=parent_data.get("DOI") or "",
        zotero_url=zotero_url(username, parent_key, parent_data.get("collections")),
        collection_name=candidate.collection_name,
        content_type=candidate.content_type,
        source_format=candidate.format,
        filename=str(data.get("filename") or ""),
        md5=str(data.get("md5") or ""),
        version=_int(data.get("version")),
    )


def _skip(data: dict[str, Any], collection_name: str, reason: SkipReason) -> SkippedEntry:
    return SkippedEntry(
        key=str(data["key"]),
        title=data.get("title") or str(data["key"]),
        reason=reason,
        collection_name=collection_name,
        content_type=str(data.get("contentType") or ""),
    )


def _int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0
