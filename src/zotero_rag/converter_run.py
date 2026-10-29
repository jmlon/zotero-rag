"""The back half of `convert`: download, convert, judge, upload (PLAN-markdown, Phase 4).

Each planned action is done in two stages. *Preparing* — download, convert,
assess — runs on a small thread pool, since anydoc releases the GIL and
downloads wait on the network. *Publishing* — the Zotero writes — runs on the
calling thread, one at a time, because Zotero rate-limits writes and pyzotero's
client is not built for concurrent use.

One failure never stops the run: every action ends as a `ConversionResult`.
Each publish is a single upload (creation) or an upload plus a note update
(regeneration), so an interrupted run leaves at worst a converted file in the
cache that was never uploaded, which the next run simply redoes.

Documents with scanned pages are held back when hosted OCR is on, so the user
can be asked once, with the count, before anything leaves the machine.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zotero_rag.config import QualityThresholds
from zotero_rag.conversion import ConversionAction, ConversionEntry, SourceAttachment
from zotero_rag.documents import DocumentFormat
from zotero_rag.download import AttachmentFetcher, cache_path, ensure_cached
from zotero_rag.errors import ZoteroConflictError, ZoteroRagError
from zotero_rag.extraction import ExtractionError, extract_segments
from zotero_rag.markdown import Converted, NeedsOcr, OcrMode, convert, transcript
from zotero_rag.provenance import MARKDOWN_TAG, Provenance
from zotero_rag.quality import QualityReport, QualityVerdict, assess
from zotero_rag.zotero_api import (
    create_markdown_attachment,
    file_md5,
    replace_attachment_file,
    update_attachment_note,
)

logger = logging.getLogger(__name__)

TRANSCRIPT_TITLE = "Markdown transcript"
STAGING_DIRNAME = "transcripts"
_UNSAFE_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')
_MAX_STEM = 100


class Outcome(StrEnum):
    CREATED = "created"
    REGENERATED = "regenerated"
    CONFLICT = "changed in Zotero meanwhile"
    FAILED = "failed"


class ConversionResult(BaseModel):
    """What happened to one planned action."""

    entry: ConversionEntry
    outcome: Outcome
    transcript_key: str = ""
    quality: QualityReport | None = None
    images: int = 0
    empty_pages: list[int] = Field(default_factory=list)
    ocr_pages: list[int] = Field(default_factory=list)
    # Scanned pages were left without text (OCR off, declined, or unavailable).
    unread_scans: bool = False
    error: str = ""

    @property
    def title(self) -> str:
        return self.entry.title


class ConversionRun(BaseModel):
    results: list[ConversionResult] = Field(default_factory=list)
    # True when the user was asked about hosted OCR and said no.
    ocr_declined: bool = False

    @property
    def unread_scans(self) -> list[ConversionResult]:
        """Documents with scanned pages no text was recovered from."""
        return [result for result in self.results if result.unread_scans]

    @property
    def outcome_counts(self) -> dict[Outcome, int]:
        return dict(Counter(result.outcome for result in self.results).most_common())

    @property
    def quality_counts(self) -> dict[QualityVerdict, int]:
        return dict(Counter(r.quality.verdict for r in self.results if r.quality).most_common())

    def with_quality(self, verdict: QualityVerdict) -> list[ConversionResult]:
        return [r for r in self.results if r.quality and r.quality.verdict is verdict]


class _Prepared(BaseModel):
    """A converted document waiting to be published, or why it could not be."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entry: ConversionEntry
    path: Path | None = None
    converted: Converted | None = None
    quality: QualityReport | None = None
    needs_ocr: bool = False
    error: str = ""


def run_conversion(
    actions: list[ConversionEntry],
    *,
    zot: Any,
    files: AttachmentFetcher,
    cache_dir: Path,
    thresholds: QualityThresholds,
    today: date,
    ocr: OcrMode = "off",
    api_key: str = "",
    jobs: int = 4,
    confirm_ocr: Callable[[list[ConversionEntry]], bool] = lambda entries: False,
    on_done: Callable[[ConversionResult], None] = lambda result: None,
) -> ConversionRun:
    """Carry out the planned actions and report each one.

    `confirm_ocr` is asked once, after the local pass, with the documents that
    have scanned pages; only if it says yes are those sent to the OCR service.
    """
    run = ConversionRun()
    held: list[_Prepared] = []

    def finish(result: ConversionResult) -> None:
        run.results.append(result)
        on_done(result)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures: dict[Future[_Prepared], ConversionEntry] = {
            pool.submit(_prepare, entry, files, cache_dir, thresholds, "off", api_key): entry
            for entry in actions
        }
        try:
            for future in as_completed(futures):
                prepared = future.result()
                if prepared.needs_ocr and ocr == "hosted":
                    held.append(prepared)
                    continue
                finish(_publish(prepared, zot, cache_dir, today))
        except BaseException:
            # Ctrl-C: stop handing out work; what was published stays published.
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    if held:
        entries = [prepared.entry for prepared in held]
        if confirm_ocr(entries):
            for prepared in held:
                redone = _prepare(prepared.entry, files, cache_dir, thresholds, "hosted", api_key)
                finish(_publish(redone, zot, cache_dir, today))
        else:
            run.ocr_declined = True
            for prepared in held:
                finish(_publish(prepared, zot, cache_dir, today))
    return run


def _prepare(
    entry: ConversionEntry,
    files: AttachmentFetcher,
    cache_dir: Path,
    thresholds: QualityThresholds,
    ocr: OcrMode,
    api_key: str,
) -> _Prepared:
    """Download, convert and judge one source. Never raises for an expected failure."""
    source = entry.source
    assert source is not None  # actions always have a source
    try:
        path = ensure_cached(files, source.attachment_key, cache_dir, _cache_format(source))
        converted = convert(path, source.source_format, ocr=ocr, api_key=api_key)
    except NeedsOcr as exc:
        pages = ", ".join(str(page) for page in exc.pages)
        return _Prepared(entry=entry, needs_ocr=True, error=f"needs OCR (pages {pages})")
    except ZoteroRagError as exc:
        return _Prepared(entry=entry, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - one odd document must not end the run
        logger.exception("Unexpected failure converting %s", source.attachment_key)
        return _Prepared(entry=entry, error=f"{type(exc).__name__}: {exc}")

    quality = assess(
        converted.markdown,
        _baseline(path, source),
        thresholds,
        needs_ocr=converted.needs_ocr,
        ocr_pages=converted.ocr_pages,
    )
    return _Prepared(
        entry=entry,
        path=path,
        converted=converted,
        quality=quality,
        needs_ocr=bool(converted.needs_ocr),
    )


def _publish(prepared: _Prepared, zot: Any, cache_dir: Path, today: date) -> ConversionResult:
    """Upload one transcript, creating or replacing, and keep a cached copy."""
    entry = prepared.entry
    converted = prepared.converted
    if converted is None:
        return ConversionResult(
            entry=entry,
            outcome=Outcome.FAILED,
            unread_scans=prepared.needs_ocr,
            error=prepared.error,
        )
    source = entry.source
    assert source is not None

    staged = _stage(cache_dir, source, converted, today)
    record = Provenance(
        source=source.attachment_key,
        source_md5=source.md5 or file_md5(prepared.path or staged),
        md5=file_md5(staged),
        converter=converted.converter,
        quality=prepared.quality.summary() if prepared.quality else "",
    )
    result = ConversionResult(
        entry=entry,
        outcome=Outcome.FAILED,
        quality=prepared.quality,
        images=converted.images,
        empty_pages=converted.empty_pages,
        ocr_pages=converted.ocr_pages,
        unread_scans=prepared.needs_ocr,
    )
    try:
        if entry.action is ConversionAction.CREATE:
            uploaded = create_markdown_attachment(
                zot,
                source.item_key,
                staged,
                title=TRANSCRIPT_TITLE,
                note=record.to_note(),
                tags=[MARKDOWN_TAG],
            )
            result.outcome = Outcome.CREATED
        else:
            existing = entry.transcript
            assert existing is not None  # REGENERATE always targets a transcript
            uploaded = replace_attachment_file(zot, existing.key, staged, expected_md5=existing.md5)
            try:
                update_attachment_note(zot, uploaded.key, uploaded.version, record.to_note())
            except ZoteroRagError as exc:
                # The new file is in place but its note still records the old
                # md5, so the next run will see it as edited and leave it alone.
                raise ZoteroRagError(
                    f"the file was replaced but its provenance note was not updated ({exc}); "
                    "the next run will treat it as edited — rerun with --force"
                ) from exc
            result.outcome = Outcome.REGENERATED
    except ZoteroConflictError as exc:
        result.outcome, result.error = Outcome.CONFLICT, str(exc)
        return result
    except ZoteroRagError as exc:
        result.error = str(exc)
        return result

    result.transcript_key = uploaded.key
    # Cached under the transcript's own key, so `index` reading it later is a hit.
    target = cache_path(cache_dir, uploaded.key, DocumentFormat.TEXT)
    shutil.copyfile(staged, target)
    shutil.rmtree(staged.parent, ignore_errors=True)
    return result


def _stage(cache_dir: Path, source: SourceAttachment, converted: Converted, today: date) -> Path:
    """Write the transcript to upload. Its file name is what Zotero will show."""
    folder = cache_dir / STAGING_DIRNAME / source.attachment_key
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / transcript_filename(source)
    path.write_text(
        transcript(converted, source, source_md5=source.md5, generated=today), encoding="utf-8"
    )
    return path


def transcript_filename(source: SourceAttachment) -> str:
    """`paper.pdf` → `paper.md`; falls back to the title, then the key."""
    stem = Path(source.filename).stem if source.filename else source.title
    stem = _UNSAFE_FILENAME.sub(" ", stem).strip(" .")[:_MAX_STEM].strip()
    return f"{stem or source.attachment_key}.md"


def _cache_format(source: SourceAttachment) -> DocumentFormat | str:
    return source.document_format or source.source_format


def _baseline(path: Path, source: SourceAttachment) -> list[Any] | None:
    """The independent reading quality is judged against, if the format has one."""
    document_format = source.document_format
    if document_format is None:
        return None
    try:
        return extract_segments(path, document_format)
    except ExtractionError as exc:
        logger.warning("No baseline for %s: %s", source.attachment_key, exc)
        return None


REPORT_FIELDS = (
    "title",
    "item_key",
    "source_key",
    "source_format",
    "state",
    "action",
    "outcome",
    "transcript_key",
    "quality",
    "quality_reasons",
    "images",
    "empty_pages",
    "ocr_pages",
    "error",
)


def report_rows(
    entries: list[ConversionEntry], run: ConversionRun | None
) -> list[dict[str, str | int]]:
    """One flat row per plan entry, joined with what the run did to it.

    Flat on purpose: the same rows are written as CSV for a spreadsheet and as
    JSON for scripts.
    """
    # Only actions have results, and every action has a source.
    by_source = {
        result.entry.source.attachment_key: result
        for result in (run.results if run else [])
        if result.entry.source
    }
    rows: list[dict[str, str | int]] = []
    for entry in entries:
        result = by_source.get(entry.source.attachment_key) if entry.source else None
        source = entry.source
        quality = result.quality if result else None
        rows.append(
            {
                "title": entry.title,
                "item_key": source.item_key if source else "",
                "source_key": source.attachment_key if source else "",
                "source_format": source.source_format if source else "",
                "state": str(entry.state),
                "action": str(entry.action),
                "outcome": str(result.outcome) if result else "",
                "transcript_key": (
                    result.transcript_key
                    if result and result.transcript_key
                    else (entry.transcript.key if entry.transcript else "")
                ),
                "quality": str(quality.verdict) if quality else "",
                "quality_reasons": "; ".join(quality.reasons) if quality else "",
                "images": result.images if result else 0,
                "empty_pages": _page_list(result.empty_pages) if result else "",
                "ocr_pages": _page_list(result.ocr_pages) if result else "",
                "error": result.error if result else "",
            }
        )
    return rows


def _page_list(pages: list[int]) -> str:
    return " ".join(str(page) for page in pages)
