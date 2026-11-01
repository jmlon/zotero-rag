"""Which stored files `shrink-storage` may move to the trash (PLAN-markdown, Phase 5).

A file goes only when everything below holds, and the first check that fails is
the reason it is kept:

1. It has a Markdown transcript that is up to date and passed the quality check
   (D4). A transcript the user attached is judged against the file at run time.
2. It carries no annotations: deleting a PDF deletes its highlights (R11).
3. Its item has a web link attachment (D5b) …
4. … and the file is larger than `--max-size` …
5. … and the link serves the same file: a 2xx answer, the same content type, the
   right magic bytes, and exactly the same size (R10, D5). A link that passes
   all but the size is reported under *Check manually*, never trashed.

The checks that need nothing but the item listings come first, so a file that
cannot be shrunk anyway is never downloaded just to learn its size.

Everything here is pure: downloads, probes and writes live in `shrink_run`.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from zotero_rag.conversion import (
    STORED_MODES,
    ConversionPlan,
    ConversionState,
    SourceAttachment,
    Transcript,
)
from zotero_rag.quality import QualityVerdict
from zotero_rag_core.planning import LINKED_URL_MODE

_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(i?b)?\s*$", re.IGNORECASE)
_UNITS = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4}

# What a file of each format starts with. Formats missing here are not checked
# for magic, only for content type and size.
_ZIP_MAGIC = b"PK\x03\x04"
MAGIC: dict[str, bytes] = {
    "pdf": b"%PDF",
    "docx": _ZIP_MAGIC,
    "odt": _ZIP_MAGIC,
    "epub": _ZIP_MAGIC,
    "pptx": _ZIP_MAGIC,
    "xlsx": _ZIP_MAGIC,
    "ods": _ZIP_MAGIC,
    "odp": _ZIP_MAGIC,
}
# A PDF may have bytes before its header, within the first kilobyte.
_PDF_HEADER_WINDOW = 1024


def parse_size(text: str) -> int:
    """`500KB`, `20MB`, `1.5GB` → bytes. Units are binary (1 KB = 1024 bytes).

    `KiB` and a bare number of bytes are accepted too. Raises `ValueError`.
    """
    match = _SIZE.match(text)
    if not match:
        raise ValueError(f"{text!r} is not a size; write it like 500KB, 20MB or 1.5GB")
    number, unit, _suffix = match.groups()
    size = int(float(number) * 1024 ** _UNITS[unit.lower()])
    if size <= 0:
        raise ValueError("the size must be larger than zero")
    return size


def format_size(size: int) -> str:
    """Bytes as the user wrote them: binary units, one decimal."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{size} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


class KeepReason(StrEnum):
    NO_TRANSCRIPT = "no Markdown transcript"
    TRANSCRIPT_NOT_CURRENT = "transcript not up to date"
    QUALITY = "transcript did not pass the quality check"
    ANNOTATED = "has annotations"
    NO_LINK = "no web link attachment"
    BELOW_SIZE = "not larger than --max-size"
    UNREADABLE = "could not be downloaded or read"
    PROBE_FAILED = "web link does not serve this file"
    SIZE_DIFFERS = "web link's size differs or is unknown"
    DECLINED = "declined at the prompt"
    WRITE_ERROR = "could not be moved to the trash"


class Verdict(StrEnum):
    TRASH = "move to trash"
    # A link serves a file of the right kind, but its size does not prove it is
    # the same file. The one kept reason that asks the user to act.
    CHECK = "check manually"
    KEEP = "keep"


class WebLink(BaseModel):
    """A `linked_url` attachment: the only kind of link that counts (D5b)."""

    key: str
    url: str
    title: str = ""


class ShrinkCandidate(BaseModel):
    """A file `convert` would transcribe, with what decides whether it can go."""

    source: SourceAttachment
    state: ConversionState
    # Our transcript of it, if any.
    transcript: Transcript | None = None
    user_markdown: list[Transcript] = Field(default_factory=list)
    annotated: bool = False
    links: list[WebLink] = Field(default_factory=list)

    @property
    def stored_user_markdown(self) -> list[Transcript]:
        """The user's Markdown files the API can serve; a linked file it cannot."""
        return [t for t in self.user_markdown if t.link_mode in STORED_MODES]


class ShrinkEntry(BaseModel):
    """One candidate as it moves through the checks and, perhaps, the trash."""

    candidate: ShrinkCandidate
    verdict: Verdict = Verdict.KEEP
    reason: KeepReason | None = None
    detail: str = ""
    size: int | None = None
    # The link the file can be recovered from (TRASH), or the one to check (CHECK).
    link: str = ""
    remote_size: int | None = None
    path: Path | None = None
    # Set by a real run.
    trashed: bool = False
    backup: Path | None = None
    error: str = ""

    @property
    def title(self) -> str:
        return self.candidate.source.title

    def keep(self, reason: KeepReason, detail: str = "") -> ShrinkEntry:
        self.verdict, self.reason, self.detail = Verdict.KEEP, reason, detail
        return self


def shrink_candidates(
    plan: ConversionPlan,
    collections: list[tuple[str, list[dict[str, Any]]]],
    annotated: set[str],
) -> list[ShrinkCandidate]:
    """Every source the conversion plan chose, with its links and annotations.

    Only chosen sources can have a transcript of ours, so an attachment `convert`
    passed over (a snapshot beside its PDF) is not a candidate at all.
    """
    links = _web_links(collections)
    candidates: list[ShrinkCandidate] = []
    seen: set[str] = set()
    for entry in plan.entries:
        source = entry.source
        if source is None or source.attachment_key in seen:
            continue
        seen.add(source.attachment_key)
        candidates.append(
            ShrinkCandidate(
                source=source,
                state=entry.state,
                transcript=entry.transcript,
                user_markdown=entry.user_markdown,
                annotated=source.attachment_key in annotated,
                links=links.get(source.item_key, []),
            )
        )
    return candidates


def _web_links(collections: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, list[WebLink]]:
    """Link attachments by parent item, each once however many collections list it."""
    links: dict[str, list[WebLink]] = {}
    seen: set[str] = set()
    for _name, items in collections:
        for item in items:
            data = item.get("data") or {}
            key = str(data.get("key") or item.get("key") or "")
            parent = data.get("parentItem")
            url = str(data.get("url") or "").strip()
            if (
                data.get("itemType") != "attachment"
                or data.get("linkMode") != LINKED_URL_MODE
                or not parent
                or not url.lower().startswith(("http://", "https://"))
                or key in seen
            ):
                continue
            seen.add(key)
            links.setdefault(str(parent), []).append(
                WebLink(key=key, url=url, title=str(data.get("title") or ""))
            )
    return links


def listing_reason(candidate: ShrinkCandidate) -> tuple[KeepReason, str] | None:
    """The first reason to keep a file that the item listings alone can tell."""
    if candidate.state is ConversionState.USER_PROVIDED:
        # Their Markdown is judged against this file later, once it is downloaded.
        if not candidate.stored_user_markdown:
            return KeepReason.NO_TRANSCRIPT, "your Markdown is a linked file, which can't be read"
    elif candidate.transcript is None:
        return KeepReason.NO_TRANSCRIPT, "run `zotero-rag convert` first"
    elif candidate.state is not ConversionState.CURRENT:
        return KeepReason.TRANSCRIPT_NOT_CURRENT, candidate.state.value
    else:
        record = candidate.transcript.provenance
        verdict = record.quality if record else ""
        if verdict.split(" ", 1)[0] != QualityVerdict.PASS:
            return KeepReason.QUALITY, verdict or "no verdict recorded"
    if candidate.annotated:
        return KeepReason.ANNOTATED, "deleting the file would delete its annotations"
    if not candidate.links:
        return KeepReason.NO_LINK, ""
    return None


# --- The web link check -------------------------------------------------------------


class Probe(BaseModel):
    """What a web link answered, as far as the check needs it."""

    url: str
    status: int | None = None
    content_type: str = ""
    size: int | None = None
    # The first bytes of the body, for the magic check.
    head: bytes = b""
    error: str = ""


class LinkCheck(StrEnum):
    PASS = "pass"
    SIZE = "size differs or unknown"
    FAIL = "fail"


def judge_probe(probe: Probe, source: SourceAttachment, local_size: int) -> tuple[LinkCheck, str]:
    """Does this link serve the same file? The answer and, unless PASS, why not."""
    if probe.error:
        return LinkCheck.FAIL, probe.error
    if probe.status is None or not 200 <= probe.status < 300:
        return LinkCheck.FAIL, f"HTTP {probe.status}"
    expected = _base_type(source.content_type)
    served = _base_type(probe.content_type)
    if served != expected:
        return LinkCheck.FAIL, f"serves {served or 'no content type'}, not {expected}"
    magic = MAGIC.get(source.source_format)
    if magic is not None and not _has_magic(probe.head, magic, source.source_format):
        return LinkCheck.FAIL, f"its content is not a {source.source_format} file"
    if probe.size is None:
        return LinkCheck.SIZE, "the server does not say how large the file is"
    if probe.size != local_size:
        return LinkCheck.SIZE, f"{probe.size} bytes there, {local_size} here"
    return LinkCheck.PASS, ""


def _base_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _has_magic(head: bytes, magic: bytes, source_format: str) -> bool:
    if source_format == "pdf":
        return magic in head[:_PDF_HEADER_WINDOW]
    return head.startswith(magic)


def decide_link(
    entry: ShrinkEntry, checks: list[tuple[WebLink, Probe, LinkCheck, str]]
) -> ShrinkEntry:
    """Trash if any link passes; check manually if one only fails on size; else keep."""
    for link, _probe, result, _why in checks:
        if result is LinkCheck.PASS:
            entry.verdict, entry.reason, entry.link = Verdict.TRASH, None, link.url
            return entry
    for link, probe, result, why in checks:
        if result is LinkCheck.SIZE:
            entry.verdict, entry.reason, entry.detail = Verdict.CHECK, KeepReason.SIZE_DIFFERS, why
            entry.link, entry.remote_size = link.url, probe.size
            return entry
    failures = "; ".join(f"{link.url}: {why}" for link, _probe, _result, why in checks)
    return entry.keep(KeepReason.PROBE_FAILED, failures)
