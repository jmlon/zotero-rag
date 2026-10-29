"""Phase 4 of PLAN-markdown: carrying out a conversion plan, and the `convert` command.

The Zotero side is the recording stand-in from the Phase 2 tests; conversion is
the real anydoc on fixture files. Hosted OCR is faked, as everywhere.
"""

from __future__ import annotations

import csv
import itertools
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.conftest import make_pdf_bytes
from tests.test_markdown import FakeHostedAnydoc, scanned_pdf_bytes
from tests.test_zotero_writes import FakeZotero
from zotero_rag import cli, markdown
from zotero_rag.config import QualityThresholds
from zotero_rag.conversion import (
    ConversionAction,
    ConversionEntry,
    ConversionState,
    SourceAttachment,
    Transcript,
)
from zotero_rag.converter_run import (
    REPORT_FIELDS,
    Outcome,
    report_rows,
    run_conversion,
    transcript_filename,
)
from zotero_rag.errors import ZoteroApiError
from zotero_rag.provenance import MARKDOWN_TAG, Provenance, parse
from zotero_rag.quality import QualityVerdict
from zotero_rag.zotero_api import file_md5

TODAY = date(2026, 10, 29)
# 343 distinct six-letter pseudo-words, three to a line so each fits the 200pt
# fixture page: anydoc, rightly, ignores text drawn off the page.
VOCABULARY = [
    "".join(parts)
    for parts in itertools.product(("ka", "lo", "mi", "ne", "ru", "si", "to"), repeat=3)
]
PAGES = [" ".join(VOCABULARY[i : i + 3]) for i in range(0, 240, 3)]


def paper_pdf() -> bytes:
    """Enough text, on enough pages, for a real quality verdict."""
    return make_pdf_bytes(PAGES)


class FakeFiles:
    """Stands in for `FileDownloader`: attachment key → bytes."""

    def __init__(self, contents: dict[str, bytes]) -> None:
        self.contents = contents
        self.fetched: list[str] = []

    def file(self, itemkey: str, /) -> bytes:
        self.fetched.append(itemkey)
        if itemkey not in self.contents:
            raise ZoteroApiError(f"no file for {itemkey}")
        return self.contents[itemkey]


def source(key: str = "PDFKEY01", fmt: str = "pdf", **changes: Any) -> SourceAttachment:
    values: dict[str, Any] = {
        "attachment_key": key,
        "item_key": "ITEM0001",
        "title": "A Paper",
        "authors": "Doe, J.",
        "year": "2021",
        "collection_name": "Library",
        "content_type": {"pdf": "application/pdf", "html": "text/html"}[fmt],
        "source_format": fmt,
        "filename": "paper.pdf" if fmt == "pdf" else "page.html",
        "md5": "a" * 32,
    }
    values.update(changes)
    return SourceAttachment.model_validate(values)


def create(key: str = "PDFKEY01", **changes: Any) -> ConversionEntry:
    return ConversionEntry(
        state=ConversionState.MISSING,
        action=ConversionAction.CREATE,
        source=source(key, **changes),
    )


def run(entries: list[ConversionEntry], files: dict[str, bytes], tmp_path: Path, **opts: Any):
    zot = opts.pop("zot", None) or FakeZotero()
    result = run_conversion(
        entries,
        zot=zot,
        files=FakeFiles(files),
        cache_dir=tmp_path / "cache",
        thresholds=QualityThresholds(),
        today=TODAY,
        jobs=2,
        **opts,
    )
    return result, zot


# --- Creating -----------------------------------------------------------------------


def test_a_new_transcript_is_uploaded_with_provenance(tmp_path: Path) -> None:
    result, zot = run([create()], {"PDFKEY01": paper_pdf()}, tmp_path)

    (done,) = result.results
    assert done.outcome is Outcome.CREATED
    sent = zot.uploads[0]
    assert sent["filename"] == "paper.md"
    assert sent["parentItem"] == "ITEM0001"
    assert sent["tags"] == [{"tag": MARKDOWN_TAG}]
    record = parse(sent["note"])
    assert record is not None
    assert record.source == "PDFKEY01"
    assert record.source_md5 == "a" * 32  # Zotero's md5, not the cached file's
    assert record.md5 == zot.stored[done.transcript_key]["md5"]
    assert record.quality.startswith("pass recall=")
    assert done.quality is not None and done.quality.verdict is QualityVerdict.PASS


def test_the_transcript_is_cached_for_index_and_staging_is_cleaned(tmp_path: Path) -> None:
    result, _ = run([create()], {"PDFKEY01": paper_pdf()}, tmp_path)

    key = result.results[0].transcript_key
    cached = tmp_path / "cache" / f"{key}.md"
    text = cached.read_text()
    assert text.startswith("---\ntitle: A Paper\n")
    assert "<!-- page 1 -->" in text
    assert list((tmp_path / "cache" / "transcripts").iterdir()) == []


def test_a_snapshot_is_converted_too(tmp_path: Path) -> None:
    html = f"<html><head><title>Page</title></head><body><p>{' '.join(PAGES)}</p></body></html>"

    result, _ = run([create("SNAPSHT1", fmt="html")], {"SNAPSHT1": html.encode()}, tmp_path)

    (done,) = result.results
    assert done.outcome is Outcome.CREATED
    assert done.quality is not None and done.quality.verdict is QualityVerdict.PASS


def test_one_failure_does_not_stop_the_others(tmp_path: Path) -> None:
    result, _ = run(
        [create("PDFKEY01"), create("MISSING1"), create("PDFKEY02")],
        {"PDFKEY01": paper_pdf(), "PDFKEY02": paper_pdf()},
        tmp_path,
    )

    assert result.outcome_counts == {Outcome.CREATED: 2, Outcome.FAILED: 1}
    (failed,) = [r for r in result.results if r.outcome is Outcome.FAILED]
    assert "MISSING1" in failed.error


# --- Regenerating -------------------------------------------------------------------


def existing(zot: FakeZotero, tmp_path: Path) -> Transcript:
    """A transcript already in Zotero, uploaded by an earlier run."""
    old = tmp_path / "old.md"
    old.write_text("old transcript\n")
    md5 = file_md5(old)
    note = Provenance(source="PDFKEY01", source_md5="9" * 32, md5=md5, converter="x").to_note()
    zot.stored["MDKEY001"] = {"key": "MDKEY001", "md5": md5, "version": 5, "note": note}
    return Transcript(key="MDKEY001", version=5, md5=md5, provenance=parse(note))


def test_a_stale_transcript_is_replaced_and_its_note_updated(tmp_path: Path) -> None:
    zot = FakeZotero()
    entry = ConversionEntry(
        state=ConversionState.STALE,
        action=ConversionAction.REGENERATE,
        source=source(),
        transcript=existing(zot, tmp_path),
    )

    result, _ = run([entry], {"PDFKEY01": paper_pdf()}, tmp_path, zot=zot)

    (done,) = result.results
    assert done.outcome is Outcome.REGENERATED
    assert done.transcript_key == "MDKEY001"
    record = parse(zot.stored["MDKEY001"]["note"])
    assert record is not None
    assert record.md5 == zot.stored["MDKEY001"]["md5"]  # the note matches the new file
    assert record.source_md5 == "a" * 32


def test_a_transcript_edited_during_the_run_is_a_conflict_not_overwritten(
    tmp_path: Path,
) -> None:
    zot = FakeZotero()
    transcript = existing(zot, tmp_path)
    zot.stored["MDKEY001"]["md5"] = "e" * 32  # edited after the plan was made
    entry = ConversionEntry(
        state=ConversionState.STALE,
        action=ConversionAction.REGENERATE,
        source=source(),
        transcript=transcript,
    )

    result, _ = run([entry], {"PDFKEY01": paper_pdf()}, tmp_path, zot=zot)

    assert result.results[0].outcome is Outcome.CONFLICT
    assert zot.stored["MDKEY001"]["md5"] == "e" * 32


def test_a_note_that_fails_after_the_file_was_replaced_says_what_to_do(
    tmp_path: Path,
) -> None:
    zot = FakeZotero()
    entry = ConversionEntry(
        state=ConversionState.STALE,
        action=ConversionAction.REGENERATE,
        source=source(),
        transcript=existing(zot, tmp_path),
    )

    def broken_note(payload: dict[str, Any]) -> bool:
        raise ZoteroApiError("network down")

    zot.update_item = broken_note  # type: ignore[method-assign]

    result, _ = run([entry], {"PDFKEY01": paper_pdf()}, tmp_path, zot=zot)

    (done,) = result.results
    assert done.outcome is Outcome.FAILED
    assert "rerun with --force" in done.error


# --- Scanned pages and OCR ----------------------------------------------------------


def test_a_scan_without_ocr_fails_and_is_counted(tmp_path: Path) -> None:
    result, zot = run([create()], {"PDFKEY01": scanned_pdf_bytes([None, None])}, tmp_path)

    (done,) = result.results
    assert done.outcome is Outcome.FAILED
    assert done.error.startswith("needs OCR (pages 1, 2)")
    assert result.unread_scans == [done]
    assert zot.uploads == []


def test_a_partial_scan_without_ocr_is_uploaded_but_fails_quality(tmp_path: Path) -> None:
    result, _ = run([create()], {"PDFKEY01": scanned_pdf_bytes([*PAGES, None])}, tmp_path)

    (done,) = result.results
    assert done.outcome is Outcome.CREATED
    assert done.unread_scans
    assert done.quality is not None and done.quality.verdict is QualityVerdict.FAIL


def test_hosted_ocr_asks_once_and_sends_only_after_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeHostedAnydoc()
    monkeypatch.setattr(markdown, "_anydoc", lambda: fake)
    asked: list[list[str]] = []

    def yes(entries: list[ConversionEntry]) -> bool:
        asked.append(sorted(e.source.attachment_key for e in entries if e.source))
        return True

    result, _ = run(
        [create("SCANNED1"), create("PARTIAL1"), create("TEXTONLY")],
        {
            "SCANNED1": scanned_pdf_bytes([None]),
            "PARTIAL1": scanned_pdf_bytes([*PAGES, None]),
            "TEXTONLY": paper_pdf(),
        },
        tmp_path,
        ocr="hosted",
        confirm_ocr=yes,
    )

    assert asked == [["PARTIAL1", "SCANNED1"]]
    assert len(fake.hosted) == 2  # one page from each, nothing from TEXTONLY
    by_key = {r.entry.source.attachment_key: r for r in result.results if r.entry.source}
    assert by_key["SCANNED1"].outcome is Outcome.CREATED
    assert by_key["SCANNED1"].ocr_pages == [1]
    assert by_key["PARTIAL1"].ocr_pages == [len(PAGES) + 1]
    # Text from OCR has nothing independent to be checked against.
    assert by_key["PARTIAL1"].quality.verdict is QualityVerdict.UNVERIFIED  # type: ignore[union-attr]
    assert result.unread_scans == []


def test_declining_ocr_keeps_everything_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeHostedAnydoc()
    monkeypatch.setattr(markdown, "_anydoc", lambda: fake)

    result, _ = run(
        [create("SCANNED1"), create("PARTIAL1")],
        {"SCANNED1": scanned_pdf_bytes([None]), "PARTIAL1": scanned_pdf_bytes([*PAGES, None])},
        tmp_path,
        ocr="hosted",
        confirm_ocr=lambda entries: False,
    )

    assert fake.hosted == []
    assert result.ocr_declined
    assert {r.outcome for r in result.results} == {Outcome.CREATED, Outcome.FAILED}
    assert len(result.unread_scans) == 2


# --- Naming and reporting -----------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "title", "expected"),
    [
        ("paper.pdf", "T", "paper.md"),
        ("", "Colons: and / slashes?", "Colons  and   slashes.md"),
        ("", "", "PDFKEY01.md"),
        ("x" * 300 + ".pdf", "T", "x" * 100 + ".md"),
    ],
)
def test_transcript_filename(filename: str, title: str, expected: str) -> None:
    assert transcript_filename(source(filename=filename, title=title)) == expected


def test_report_rows_join_the_plan_and_the_run(tmp_path: Path) -> None:
    entries = [
        create(),
        ConversionEntry(state=ConversionState.CURRENT, source=source("PDFKEY02")),
    ]
    result, _ = run(entries[:1], {"PDFKEY01": paper_pdf()}, tmp_path)

    rows = report_rows(entries, result)

    assert [set(row) for row in rows] == [set(REPORT_FIELDS)] * 2
    assert rows[0]["outcome"] == "created"
    assert rows[0]["quality"] == "pass"
    assert rows[1]["state"] == "up to date"
    assert rows[1]["outcome"] == ""


# --- The command ----------------------------------------------------------------------

runner = CliRunner()


def _library(monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]]) -> FakeZotero:
    zot = FakeZotero()
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr(cli, "build_client", lambda config: zot)
    monkeypatch.setattr(
        cli,
        "fetch_collections",
        lambda z: [{"key": "COLL0001", "data": {"key": "COLL0001", "name": "Papers"}}],
    )
    monkeypatch.setattr(cli, "fetch_username", lambda z: "someone")
    monkeypatch.setattr(cli, "fetch_collection_items", lambda z, key: items)
    monkeypatch.setenv("COLUMNS", "200")
    return zot


def _items(*keys: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for number, key in enumerate(keys, start=1):
        parent = f"ITEM{number:04d}"
        items.append(
            {
                "key": parent,
                "data": {"key": parent, "itemType": "journalArticle", "title": f"Paper {number}"},
            }
        )
        items.append(
            {
                "key": key,
                "data": {
                    "key": key,
                    "itemType": "attachment",
                    "parentItem": parent,
                    "contentType": "application/pdf",
                    "linkMode": "imported_file",
                    "filename": f"paper{number}.pdf",
                    "md5": "a" * 32,
                    "version": 3,
                },
            }
        )
    return items


@pytest.fixture
def index_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "config.yaml").write_text("index:\n  collection_key: COLL0001\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_dry_run_lists_and_writes_nothing(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zot = _library(monkeypatch, _items("PDFKEY01", "PDFKEY02"))

    result = runner.invoke(cli.app, ["convert", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Transcripts that would be written" in result.output
    assert "2 transcripts would be written" in result.output
    assert "Dry run: nothing was downloaded, converted, or written." in result.output
    assert zot.uploads == []


def test_a_real_run_converts_uploads_and_reports(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zot = _library(monkeypatch, _items("PDFKEY01", "PDFKEY02"))
    files = FakeFiles({"PDFKEY01": paper_pdf(), "PDFKEY02": paper_pdf()})
    monkeypatch.setattr(cli.FileDownloader, "from_config", lambda config: files)

    result = runner.invoke(cli.app, ["convert", "--yes", "--report", "out.csv"])

    assert result.exit_code == 0, result.output
    assert "Wrote 2 transcripts (2 new, 0 regenerated)" in result.output
    assert "quality: 2 pass" in result.output
    assert len(zot.uploads) == 2
    with (index_folder / "out.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["outcome"] for row in rows] == ["created", "created"]


def test_limit_and_only_item_narrow_the_run(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _library(monkeypatch, _items("PDFKEY01", "PDFKEY02", "PDFKEY03"))
    files = FakeFiles({k: paper_pdf() for k in ("PDFKEY01", "PDFKEY02", "PDFKEY03")})
    monkeypatch.setattr(cli.FileDownloader, "from_config", lambda config: files)

    runner.invoke(cli.app, ["convert", "--yes", "--only-item", "item0002"])
    assert files.fetched == ["PDFKEY02"]

    result = runner.invoke(cli.app, ["convert", "--dry-run", "--limit", "1"])
    assert "1 transcript would be written" in result.output
    assert "2 more left for a later run" in result.output


def test_a_read_only_key_stops_before_any_download(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zot = _library(monkeypatch, _items("PDFKEY01"))
    zot.access["user"]["write"] = False
    files = FakeFiles({})
    monkeypatch.setattr(cli.FileDownloader, "from_config", lambda config: files)

    result = runner.invoke(cli.app, ["convert", "--yes"])

    assert result.exit_code == 1
    assert "write access" in result.output
    assert files.fetched == []


def test_declining_the_write_confirmation_writes_nothing(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zot = _library(monkeypatch, _items("PDFKEY01"))
    monkeypatch.setattr(cli.FileDownloader, "from_config", lambda config: FakeFiles({}))

    result = runner.invoke(cli.app, ["convert"], input="n\n")

    assert result.exit_code != 0
    assert "Write 1 transcript to Zotero (1 new, 0 regenerated)?" in result.output
    assert zot.uploads == []


def test_an_up_to_date_library_needs_nothing(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = _items("PDFKEY01")
    note = Provenance(
        source="PDFKEY01",
        source_md5="a" * 32,
        md5="c" * 32,
        converter=markdown.converter_name("pdf"),
    ).to_note()
    items.append(
        {
            "key": "MDKEY001",
            "data": {
                "key": "MDKEY001",
                "itemType": "attachment",
                "parentItem": "ITEM0001",
                "contentType": "text/markdown",
                "linkMode": "imported_file",
                "filename": "paper1.md",
                "md5": "c" * 32,
                "note": note,
            },
        }
    )
    _library(monkeypatch, items)

    result = runner.invoke(cli.app, ["convert", "--yes", "--report", "r.json"])

    assert result.exit_code == 0, result.output
    assert "Nothing to convert" in result.output
    rows = json.loads((index_folder / "r.json").read_text())
    assert rows[0]["state"] == "up to date"


def test_ocr_must_be_off_or_hosted(index_folder: Path) -> None:
    result = runner.invoke(cli.app, ["convert", "--ocr", "local"])

    assert result.exit_code == 1
    assert "--ocr must be off or hosted" in result.output
