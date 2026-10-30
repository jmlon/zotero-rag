"""The I/O half of `shrink-storage`: download, probe, back up, trash (PLAN-markdown, Phase 5).

Checking runs on a small thread pool, because it is all waiting on the network:
downloading a file to learn its size, and asking its web link about it. The
link prober allows two requests at a time per host, to stay polite to the
publishers' servers. Trashing runs on the calling thread, one item at a time.

Nothing is ever deleted. A trashed file stays in the Zotero trash until the user
empties it, and by default a copy is kept in `<index>/shrink-backup/` too.
"""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from zotero_rag import __version__
from zotero_rag.config import QualityThresholds
from zotero_rag.documents import DocumentFormat
from zotero_rag.download import AttachmentFetcher, ensure_cached
from zotero_rag.errors import ZoteroRagError
from zotero_rag.extraction import ExtractionError, extract_segments
from zotero_rag.markdown import body
from zotero_rag.quality import QualityVerdict, assess
from zotero_rag.shrink import (
    KeepReason,
    Probe,
    ShrinkCandidate,
    ShrinkEntry,
    Verdict,
    decide_link,
    judge_probe,
    listing_reason,
)
from zotero_rag.zotero_api import file_md5, trash_item, update_attachment_note

logger = logging.getLogger(__name__)

BACKUP_DIRNAME = "shrink-backup"
_PROBE_BYTES = 2048
_PER_HOST = 2

ProbeFn = Callable[[str], Probe]


class LinkProber:
    """Asks a web link what it serves, without downloading the file.

    A HEAD request first; many publishers refuse it (403, 405), so a GET for the
    first 2 KB follows either way, and is what the magic check reads. Its
    `Content-Range` total, or failing that a `Content-Length`, is the size.
    `Accept-Encoding: identity`, so a size is never that of a compressed copy.
    """

    def __init__(self, client: httpx.Client | None = None, per_host: int = _PER_HOST) -> None:
        self._client = client or httpx.Client(
            timeout=10.0,
            follow_redirects=True,
            headers={
                "User-Agent": f"zotero-rag/{__version__} (checking a link before freeing storage)",
                "Accept-Encoding": "identity",
            },
        )
        self._per_host = per_host
        self._hosts: dict[str, threading.Semaphore] = {}
        self._lock = threading.Lock()

    def __call__(self, url: str) -> Probe:
        with self._slot(urlsplit(url).hostname or ""):
            try:
                return self._probe(url)
            except httpx.HTTPError as exc:
                return Probe(url=url, error=f"{type(exc).__name__}: {exc}")

    def _slot(self, host: str) -> threading.Semaphore:
        with self._lock:
            return self._hosts.setdefault(host, threading.Semaphore(self._per_host))

    def _probe(self, url: str) -> Probe:
        head_size: int | None = None
        try:
            head = self._client.head(url)
            # Only a whole-file answer gives the file's length; a 206 gives a part's.
            if head.status_code == 200:
                head_size = _int_header(head.headers.get("Content-Length"))
        except httpx.HTTPError as exc:
            logger.debug("HEAD %s failed: %s", url, exc)

        ranged = {"Range": f"bytes=0-{_PROBE_BYTES - 1}"}
        with self._client.stream("GET", url, headers=ranged) as response:
            if not response.is_success:
                # The GET is what serves the file; a HEAD that liked it is not enough.
                return Probe(url=url, status=response.status_code)
            data = b""
            for chunk in response.iter_bytes():
                data += chunk
                if len(data) >= _PROBE_BYTES:
                    break
            if response.status_code == 206:
                size = _range_total(response.headers.get("Content-Range"))
            else:  # the server ignored the range and is sending the whole file
                size = _int_header(response.headers.get("Content-Length"))
            return Probe(
                url=url,
                status=response.status_code,
                content_type=response.headers.get("Content-Type", ""),
                size=size if size is not None else head_size,
                head=data[:_PROBE_BYTES],
            )

    def close(self) -> None:
        self._client.close()


def _int_header(value: str | None) -> int | None:
    return int(value) if value and value.strip().isdigit() else None


def _range_total(value: str | None) -> int | None:
    """`bytes 0-2047/183654` → 183654; `*` (unknown) → None."""
    if not value or "/" not in value:
        return None
    return _int_header(value.rsplit("/", 1)[1])


# --- Checking -------------------------------------------------------------------


def check_candidates(
    candidates: list[ShrinkCandidate],
    *,
    files: AttachmentFetcher,
    cache_dir: Path,
    max_size: int,
    thresholds: QualityThresholds,
    probe: ProbeFn,
    jobs: int = 4,
    on_done: Callable[[ShrinkEntry], None] = lambda entry: None,
) -> list[ShrinkEntry]:
    """Run every check on every candidate; returns entries in candidate order."""

    def check(candidate: ShrinkCandidate) -> ShrinkEntry:
        entry = _check(candidate, files, cache_dir, max_size, thresholds, probe)
        on_done(entry)
        return entry

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        try:
            return list(pool.map(check, candidates))
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def _check(
    candidate: ShrinkCandidate,
    files: AttachmentFetcher,
    cache_dir: Path,
    max_size: int,
    thresholds: QualityThresholds,
    probe: ProbeFn,
) -> ShrinkEntry:
    entry = ShrinkEntry(candidate=candidate)
    early = listing_reason(candidate)
    if early is not None:
        return entry.keep(*early)

    source = candidate.source
    try:
        entry.path = _cached(
            files, cache_dir, source.attachment_key, _format(candidate), source.md5
        )
    except ZoteroRagError as exc:
        return entry.keep(KeepReason.UNREADABLE, str(exc))
    entry.size = entry.path.stat().st_size
    if entry.size <= max_size:
        return entry.keep(KeepReason.BELOW_SIZE)

    if candidate.stored_user_markdown:
        failed = _user_markdown_reason(candidate, entry.path, files, cache_dir, thresholds)
        if failed:
            return entry.keep(KeepReason.QUALITY, failed)

    checks = []
    for link in candidate.links:
        answer = probe(link.url)
        result, why = judge_probe(answer, source, entry.size)
        checks.append((link, answer, result, why))
    return decide_link(entry, checks)


def _user_markdown_reason(
    candidate: ShrinkCandidate,
    source_path: Path,
    files: AttachmentFetcher,
    cache_dir: Path,
    thresholds: QualityThresholds,
) -> str:
    """Why none of the user's Markdown files vouches for the source; "" if one does.

    There is no stored verdict for Markdown we did not write, so it is judged
    now, against an independent reading of this very file.
    """
    document_format = candidate.source.document_format
    try:
        baseline = extract_segments(source_path, document_format) if document_format else None
    except ExtractionError as exc:
        logger.warning("No baseline for %s: %s", candidate.source.attachment_key, exc)
        baseline = None
    reasons: list[str] = []
    for markdown in candidate.stored_user_markdown:
        try:
            path = _cached(files, cache_dir, markdown.key, DocumentFormat.MARKDOWN, markdown.md5)
            text = path.read_text(encoding="utf-8", errors="replace")
        except (ZoteroRagError, OSError) as exc:
            reasons.append(f"{markdown.filename or markdown.key}: {exc}")
            continue
        report = assess(body(text), baseline, thresholds)
        if report.verdict is QualityVerdict.PASS:
            return ""
        detail = "; ".join(report.reasons)
        reasons.append(f"{markdown.filename or markdown.key}: {report.summary()} ({detail})")
    return "; ".join(reasons)


def _cached(
    files: AttachmentFetcher,
    cache_dir: Path,
    key: str,
    document_format: DocumentFormat | str,
    md5: str,
) -> Path:
    """The file as Zotero holds it now. A cached copy of an older version is replaced.

    `index` and `convert` trust any cached copy; here the size decides what is
    trashed and the copy becomes the backup, so it must be the current file.
    """
    path = ensure_cached(files, key, cache_dir, document_format)
    if md5 and file_md5(path) != md5:
        logger.info("cached copy of %s is outdated; downloading it again", key)
        path.unlink()
        # A PDF stored with an upload envelope never matches Zotero's md5 once
        # unwrapped, so this second copy is taken as it comes.
        path = ensure_cached(files, key, cache_dir, document_format)
    return path


def _format(candidate: ShrinkCandidate) -> DocumentFormat | str:
    return candidate.source.document_format or candidate.source.source_format


# --- Trashing -------------------------------------------------------------------


def trash_files(
    entries: list[ShrinkEntry],
    *,
    zot: Any,
    backup_dir: Path | None,
    today: date,
    on_done: Callable[[ShrinkEntry], None] = lambda entry: None,
) -> None:
    """Back up, trash, and record on the transcript, for each entry to trash.

    One failure never stops the run. An entry that could not be trashed is kept
    with the reason; one trashed whose transcript note could not be updated is
    trashed, with the error saying so.
    """
    for entry in entries:
        if entry.verdict is Verdict.TRASH:
            _trash_one(entry, zot, backup_dir, today)
            on_done(entry)


def _trash_one(entry: ShrinkEntry, zot: Any, backup_dir: Path | None, today: date) -> None:
    candidate = entry.candidate
    source = candidate.source
    assert entry.path is not None  # a file to trash was downloaded to be measured
    if backup_dir is not None:
        target = backup_dir / f"{source.attachment_key}{entry.path.suffix}"
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry.path, target)
        except OSError as exc:
            entry.keep(KeepReason.WRITE_ERROR, f"could not write the backup: {exc}")
            return
        entry.backup = target

    try:
        trash_item(zot, source.attachment_key, source.version)
    except ZoteroRagError as exc:
        entry.keep(KeepReason.WRITE_ERROR, str(exc))
        return
    entry.trashed = True

    transcript = candidate.transcript
    record = transcript.provenance if transcript else None
    if transcript is None or record is None:
        return  # the user's own Markdown: not ours to annotate
    updated = record.model_copy(update={"source_removed": today, "source_url": entry.link})
    try:
        update_attachment_note(zot, transcript.key, transcript.version, updated.to_note())
    except ZoteroRagError as exc:
        entry.error = (
            f"moved to the trash, but the transcript's note was not updated ({exc}); "
            "`convert` will report the transcript as having lost its source"
        )


# --- Report ---------------------------------------------------------------------

REPORT_FIELDS = (
    "section",
    "reason",
    "title",
    "item_key",
    "attachment_key",
    "filename",
    "size",
    "link",
    "remote_size",
    "backup",
    "transcript_key",
    "detail",
    "error",
)


def section(entry: ShrinkEntry) -> str:
    if entry.trashed:
        return "moved to trash"
    if entry.verdict is Verdict.TRASH:
        return "would move to trash"
    if entry.verdict is Verdict.CHECK:
        return "check manually"
    return "kept"


def report_rows(entries: list[ShrinkEntry]) -> list[dict[str, str | int]]:
    """One flat row per candidate, written as CSV or JSON alike."""
    rows: list[dict[str, str | int]] = []
    for entry in entries:
        candidate = entry.candidate
        source = candidate.source
        rows.append(
            {
                "section": section(entry),
                "reason": str(entry.reason) if entry.reason else "",
                "title": entry.title,
                "item_key": source.item_key,
                "attachment_key": source.attachment_key,
                "filename": source.filename,
                "size": entry.size if entry.size is not None else "",
                "link": entry.link,
                "remote_size": entry.remote_size if entry.remote_size is not None else "",
                "backup": str(entry.backup) if entry.backup else "",
                "transcript_key": candidate.transcript.key if candidate.transcript else "",
                "detail": entry.detail,
                "error": entry.error,
            }
        )
    return rows
