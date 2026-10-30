"""Phase 5 of PLAN-markdown: `shrink-storage`.

Eligibility is checked on hand-built item payloads run through the real
conversion planner; web links through httpx's MockTransport; trashing against
the recording Zotero stand-in from the Phase 2 tests.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from tests.test_converter_run import PAGES, FakeFiles, paper_pdf
from tests.test_zotero_writes import FakeZotero
from zotero_rag import cli
from zotero_rag.config import QualityThresholds
from zotero_rag.conversion import build_conversion_plan
from zotero_rag.markdown import converter_name
from zotero_rag.provenance import Provenance, parse
from zotero_rag.shrink import (
    KeepReason,
    LinkCheck,
    Probe,
    ShrinkEntry,
    Verdict,
    format_size,
    judge_probe,
    parse_size,
    shrink_candidates,
)
from zotero_rag.shrink_run import (
    LinkProber,
    check_candidates,
    report_rows,
    trash_files,
)

TODAY = date(2026, 11, 2)
URL = "https://example.org/paper.pdf"
PDF = b"%PDF-1.4\n" + b"x" * 5000
TRANSCRIPT = b"<!-- page 1 -->\n\nSome text\n"


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324


# --- Sizes ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "size"),
    [
        ("500KB", 500 * 1024),
        ("20MB", 20 * 1024**2),
        ("1.5GB", int(1.5 * 1024**3)),
        ("20 mb", 20 * 1024**2),
        ("20MiB", 20 * 1024**2),
        ("20M", 20 * 1024**2),
        ("4096", 4096),
        ("10B", 10),
    ],
)
def test_sizes_parse_in_binary_units(text: str, size: int) -> None:
    assert parse_size(text) == size


@pytest.mark.parametrize("text", ["", "MB", "20XB", "-1MB", "0", "twenty"])
def test_a_bad_size_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="size"):
        parse_size(text)


def test_sizes_print_as_they_are_written() -> None:
    assert format_size(512) == "512 B"
    assert format_size(20 * 1024**2) == "20.0 MB"
    assert format_size(3 * 1024**4) == "3072.0 GB"


# --- A small library ----------------------------------------------------------------


def note(**changes: Any) -> str:
    values: dict[str, Any] = {
        "source": "PDFKEY01",
        "source_md5": md5(PDF),
        "md5": md5(TRANSCRIPT),
        "converter": converter_name("pdf"),
        "source_format": "pdf",
        "quality": "pass recall=0.96 worst=0.93@1",
    }
    values.update(changes)
    return Provenance.model_validate(values).to_note()


def library(
    *,
    transcript: dict[str, Any] | None = None,
    links: tuple[str, ...] = (URL,),
    user_markdown: dict[str, Any] | None = None,
    with_transcript: bool = True,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [
        {
            "key": "ITEM0001",
            "data": {"key": "ITEM0001", "itemType": "journalArticle", "title": "A Paper"},
        },
        {
            "key": "PDFKEY01",
            "data": {
                "key": "PDFKEY01",
                "itemType": "attachment",
                "parentItem": "ITEM0001",
                "linkMode": "imported_file",
                "contentType": "application/pdf",
                "filename": "paper.pdf",
                "md5": md5(PDF),
                "version": 3,
            },
        },
    ]
    if with_transcript:
        data = {
            "key": "MDKEY001",
            "itemType": "attachment",
            "parentItem": "ITEM0001",
            "linkMode": "imported_file",
            "contentType": "text/markdown",
            "filename": "paper.md",
            "md5": md5(TRANSCRIPT),
            "version": 5,
            "note": note(),
        }
        data.update(transcript or {})
        items.append({"key": "MDKEY001", "data": data})
    if user_markdown is not None:
        data = {
            "key": "USERMD01",
            "itemType": "attachment",
            "parentItem": "ITEM0001",
            "linkMode": "imported_file",
            "contentType": "text/plain",
            "filename": "notes.md",
            "md5": "b" * 32,
            "version": 4,
        }
        data.update(user_markdown)
        items.append({"key": "USERMD01", "data": data})
    for number, url in enumerate(links, start=1):
        key = f"LINK{number:04d}"
        items.append(
            {
                "key": key,
                "data": {
                    "key": key,
                    "itemType": "attachment",
                    "parentItem": "ITEM0001",
                    "linkMode": "linked_url",
                    "url": url,
                    "title": "Full Text",
                },
            }
        )
    return items


def candidates(items: list[dict[str, Any]], annotated: set[str] | None = None):
    collections = [("Papers", items)]
    plan = build_conversion_plan(collections)
    return shrink_candidates(plan, collections, annotated or set())


def good_probe(url: str) -> Probe:
    return Probe(
        url=url, status=200, content_type="application/pdf", size=len(PDF), head=PDF[:2048]
    )


def check(
    items: list[dict[str, Any]],
    tmp_path: Path,
    *,
    probe: Any = good_probe,
    files: dict[str, bytes] | None = None,
    max_size: int = 1024,
    annotated: set[str] | None = None,
) -> ShrinkEntry:
    (entry,) = check_candidates(
        candidates(items, annotated),
        files=FakeFiles(files if files is not None else {"PDFKEY01": PDF}),
        cache_dir=tmp_path / "cache",
        max_size=max_size,
        thresholds=QualityThresholds(),
        probe=probe,
        jobs=2,
    )
    return entry


# --- Eligibility, one reason at a time ------------------------------------------------


def test_a_large_file_with_a_good_transcript_and_link_can_go(tmp_path: Path) -> None:
    entry = check(library(), tmp_path)

    assert entry.verdict is Verdict.TRASH
    assert entry.link == URL
    assert entry.size == len(PDF)


def test_no_transcript_keeps_the_file_without_downloading_it(tmp_path: Path) -> None:
    files = FakeFiles({"PDFKEY01": PDF})
    (entry,) = check_candidates(
        candidates(library(with_transcript=False)),
        files=files,
        cache_dir=tmp_path,
        max_size=1,
        thresholds=QualityThresholds(),
        probe=good_probe,
    )

    assert entry.reason is KeepReason.NO_TRANSCRIPT
    assert files.fetched == []


@pytest.mark.parametrize(
    ("changes", "reason", "detail"),
    [
        ({"md5": "c" * 32}, KeepReason.TRANSCRIPT_NOT_CURRENT, "transcript edited in Zotero"),
        (
            {"note": note(source_md5="d" * 32)},
            KeepReason.TRANSCRIPT_NOT_CURRENT,
            "source changed since conversion",
        ),
        (
            {"note": note(converter="anydoc 0.0.1 r0")},
            KeepReason.TRANSCRIPT_NOT_CURRENT,
            "made by an older converter",
        ),
        ({"note": note(quality="fail recall=0.50")}, KeepReason.QUALITY, "fail recall=0.50"),
        ({"note": note(quality="unverified")}, KeepReason.QUALITY, "unverified"),
        ({"note": note(quality="")}, KeepReason.QUALITY, "no verdict recorded"),
    ],
)
def test_only_a_current_passing_transcript_counts(
    tmp_path: Path, changes: dict[str, Any], reason: KeepReason, detail: str
) -> None:
    entry = check(library(transcript=changes), tmp_path)

    assert entry.verdict is Verdict.KEEP
    assert entry.reason is reason
    assert entry.detail == detail


def test_annotations_keep_the_file(tmp_path: Path) -> None:
    entry = check(library(), tmp_path, annotated={"PDFKEY01"})

    assert entry.reason is KeepReason.ANNOTATED


def test_no_link_attachment_keeps_the_file(tmp_path: Path) -> None:
    items = library(links=())
    # The item's own URL field is not a link attachment (D5b).
    items[0]["data"]["url"] = URL

    assert check(items, tmp_path).reason is KeepReason.NO_LINK


def test_a_small_file_is_kept(tmp_path: Path) -> None:
    entry = check(library(), tmp_path, max_size=len(PDF))

    assert entry.reason is KeepReason.BELOW_SIZE
    assert entry.size == len(PDF)


def test_a_file_that_cannot_be_downloaded_is_kept(tmp_path: Path) -> None:
    entry = check(library(), tmp_path, files={})

    assert entry.reason is KeepReason.UNREADABLE


def test_an_outdated_cached_copy_is_downloaded_again(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "PDFKEY01.pdf").write_bytes(b"%PDF-1.4 an older version")

    entry = check(library(), tmp_path)

    assert entry.verdict is Verdict.TRASH
    assert (cache / "PDFKEY01.pdf").read_bytes() == PDF


def test_a_failed_probe_names_every_link(tmp_path: Path) -> None:
    items = library(links=(URL, "https://example.org/landing"))

    def probe(url: str) -> Probe:
        return Probe(url=url, status=200, content_type="text/html", size=1234)

    entry = check(items, tmp_path, probe=probe)

    assert entry.reason is KeepReason.PROBE_FAILED
    assert URL in entry.detail and "landing" in entry.detail
    assert "serves text/html, not application/pdf" in entry.detail


def test_any_passing_link_is_enough(tmp_path: Path) -> None:
    items = library(links=("https://example.org/landing", URL))

    def probe(url: str) -> Probe:
        return good_probe(url) if url == URL else Probe(url=url, status=403)

    entry = check(items, tmp_path, probe=probe)

    assert entry.verdict is Verdict.TRASH
    assert entry.link == URL


@pytest.mark.parametrize("remote", [len(PDF) + 1, None])
def test_a_size_that_differs_or_is_unknown_asks_for_a_manual_check(
    tmp_path: Path, remote: int | None
) -> None:
    def probe(url: str) -> Probe:
        return good_probe(url).model_copy(update={"size": remote})

    entry = check(library(), tmp_path, probe=probe)

    assert entry.verdict is Verdict.CHECK
    assert entry.reason is KeepReason.SIZE_DIFFERS
    assert entry.link == URL
    assert entry.remote_size == remote


def test_users_markdown_is_judged_against_the_file(tmp_path: Path) -> None:
    pdf = paper_pdf()
    words = " ".join(" ".join(page.split()) for page in PAGES)
    good = f"# My notes\n\n{words}\n".encode()
    items = library(
        with_transcript=False,
        user_markdown={"md5": md5(good)},
    )
    items[1]["data"]["md5"] = md5(pdf)

    def probe(url: str) -> Probe:
        return Probe(
            url=url, status=200, content_type="application/pdf", size=len(pdf), head=pdf[:2048]
        )

    entry = check(items, tmp_path, probe=probe, files={"PDFKEY01": pdf, "USERMD01": good})
    assert entry.verdict is Verdict.TRASH

    bad = b"# My notes\n\nNothing like the paper at all.\n"
    items[-2]["data"]["md5"] = md5(bad)
    entry = check(items, tmp_path / "other", probe=probe, files={"PDFKEY01": pdf, "USERMD01": bad})
    assert entry.reason is KeepReason.QUALITY
    assert "notes.md" in entry.detail


def test_users_markdown_as_a_linked_file_cannot_vouch(tmp_path: Path) -> None:
    items = library(with_transcript=False, user_markdown={"linkMode": "linked_file"})

    entry = check(items, tmp_path)

    assert entry.reason is KeepReason.NO_TRANSCRIPT
    assert "linked file" in entry.detail


# --- Judging a link's answer ----------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "result", "why"),
    [
        ({}, LinkCheck.PASS, ""),
        ({"content_type": "application/pdf; charset=binary"}, LinkCheck.PASS, ""),
        ({"error": "ConnectTimeout: timed out"}, LinkCheck.FAIL, "timed out"),
        ({"status": 404}, LinkCheck.FAIL, "HTTP 404"),
        ({"content_type": "text/html"}, LinkCheck.FAIL, "serves text/html"),
        ({"content_type": ""}, LinkCheck.FAIL, "no content type"),
        ({"head": b"<html>paywall"}, LinkCheck.FAIL, "not a pdf file"),
        ({"size": None}, LinkCheck.SIZE, "does not say"),
        ({"size": 10}, LinkCheck.SIZE, "10 bytes there"),
    ],
)
def test_judging_a_probe(changes: dict[str, Any], result: LinkCheck, why: str) -> None:
    (candidate,) = candidates(library())
    probe = good_probe(URL).model_copy(update=changes)

    got, reason = judge_probe(probe, candidate.source, len(PDF))

    assert got is result
    assert why in reason


def test_pdf_magic_may_follow_a_few_junk_bytes() -> None:
    (candidate,) = candidates(library())
    probe = good_probe(URL).model_copy(update={"head": b"\r\n\r\n" + PDF[:100]})

    assert judge_probe(probe, candidate.source, len(PDF))[0] is LinkCheck.PASS


# --- Probing a link over HTTP ---------------------------------------------------------


def prober(handler: Any) -> LinkProber:
    return LinkProber(httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True))


def test_head_refused_then_a_ranged_get() -> None:
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.headers.get("Range")))
        if request.method == "HEAD":
            return httpx.Response(405)
        return httpx.Response(
            206,
            headers={"Content-Type": "application/pdf", "Content-Range": "bytes 0-2047/5009"},
            content=PDF[:2048],
        )

    probe = prober(handler)(URL)

    assert seen == [("HEAD", None), ("GET", "bytes=0-2047")]
    assert (probe.status, probe.content_type, probe.size) == (206, "application/pdf", 5009)
    assert probe.head.startswith(b"%PDF")


def test_a_server_ignoring_the_range_is_read_only_2kb() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "application/pdf"}, content=PDF)

    probe = prober(handler)(URL)

    assert probe.size == len(PDF)  # from Content-Length
    assert len(probe.head) == 2048


def test_redirects_are_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "doi.org":
            return httpx.Response(302, headers={"Location": "https://pub.example/a"})
        if request.url.path == "/a":
            return httpx.Response(301, headers={"Location": "https://cdn.example/paper.pdf"})
        return httpx.Response(
            206,
            headers={"Content-Type": "application/pdf", "Content-Range": "bytes 0-2047/*"},
            content=PDF[:2048],
        )

    probe = prober(handler)("https://doi.org/10.1/x")

    assert probe.status == 206
    assert probe.size is None  # "*": the server does not know


def test_head_size_is_used_when_the_get_gives_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"Content-Length": "5009"})
        return httpx.Response(206, headers={"Content-Type": "application/pdf"}, content=PDF[:10])

    assert prober(handler)(URL).size == 5009


def test_a_landing_page_and_a_dead_host() -> None:
    def landing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/html"}, content=b"<html>")

    assert prober(landing)(URL).content_type == "text/html"

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    probe = prober(dead)(URL)
    assert probe.status is None
    assert "ConnectError" in probe.error


def test_a_refused_get_is_a_failure_even_if_head_was_fine() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"Content-Type": "application/pdf"})
        return httpx.Response(403)

    assert prober(handler)(URL).status == 403


# --- Trashing -----------------------------------------------------------------------


def seeded_zotero(items: list[dict[str, Any]]) -> FakeZotero:
    zot = FakeZotero()
    for item in items:
        zot.stored[item["key"]] = dict(item["data"])
    return zot


def eligible(tmp_path: Path, items: list[dict[str, Any]]) -> ShrinkEntry:
    entry = check(items, tmp_path)
    assert entry.verdict is Verdict.TRASH
    return entry


def test_trashing_backs_up_trashes_and_records_on_the_transcript(tmp_path: Path) -> None:
    items = library()
    zot = seeded_zotero(items)
    entry = eligible(tmp_path, items)

    trash_files([entry], zot=zot, backup_dir=tmp_path / "shrink-backup", today=TODAY)

    assert entry.trashed and not entry.error
    assert entry.backup == tmp_path / "shrink-backup" / "PDFKEY01.pdf"
    assert entry.backup.read_bytes() == PDF
    assert zot.stored["PDFKEY01"]["deleted"] == 1
    record = parse(zot.stored["MDKEY001"]["note"])
    assert record is not None
    assert record.source_removed == TODAY
    assert record.source_url == URL
    assert record.quality.startswith("pass")  # the rest of the record survives


def test_trashed_source_makes_the_transcript_permanently_current(tmp_path: Path) -> None:
    items = library()
    zot = seeded_zotero(items)
    trash_files([eligible(tmp_path, items)], zot=zot, backup_dir=None, today=TODAY)

    # The next listing no longer has the source, and the transcript records why.
    after = [item for item in items if item["key"] != "PDFKEY01"]
    after[1] = {"key": "MDKEY001", "data": zot.stored["MDKEY001"]}
    plan = build_conversion_plan([("Papers", after)])

    assert [str(entry.state) for entry in plan.entries] == ["source removed by shrink-storage"]
    assert plan.actions == []


def test_no_backup_writes_no_copy(tmp_path: Path) -> None:
    items = library()
    entry = eligible(tmp_path, items)

    trash_files([entry], zot=seeded_zotero(items), backup_dir=None, today=TODAY)

    assert entry.trashed and entry.backup is None
    assert not (tmp_path / "shrink-backup").exists()


def test_a_conflict_keeps_the_file(tmp_path: Path) -> None:
    items = library()
    zot = seeded_zotero(items)
    zot.stored["PDFKEY01"]["version"] = 9  # edited in Zotero since it was listed
    entry = eligible(tmp_path, items)

    trash_files([entry], zot=zot, backup_dir=None, today=TODAY)

    assert not entry.trashed
    assert entry.reason is KeepReason.WRITE_ERROR
    assert "changed in Zotero" in entry.detail
    assert "deleted" not in zot.stored["PDFKEY01"]


def test_a_failed_note_update_is_reported_but_the_file_stays_trashed(tmp_path: Path) -> None:
    items = library()
    zot = seeded_zotero(items)
    zot.stored["MDKEY001"]["version"] = 9
    entry = eligible(tmp_path, items)

    trash_files([entry], zot=zot, backup_dir=None, today=TODAY)

    assert entry.trashed
    assert "note was not updated" in entry.error


def test_only_entries_to_trash_are_touched(tmp_path: Path) -> None:
    items = library()
    zot = seeded_zotero(items)
    entry = check(items, tmp_path, max_size=len(PDF))

    trash_files([entry], zot=zot, backup_dir=None, today=TODAY)

    assert zot.patches == []


def test_report_rows_cover_each_section(tmp_path: Path) -> None:
    items = library()
    trashed = eligible(tmp_path, items)
    trash_files([trashed], zot=seeded_zotero(items), backup_dir=None, today=TODAY)
    small = check(items, tmp_path, max_size=len(PDF))

    rows = report_rows([trashed, small])

    assert rows[0]["section"] == "moved to trash"
    assert rows[0]["link"] == URL
    assert rows[0]["transcript_key"] == "MDKEY001"
    assert rows[1]["section"] == "kept"
    assert rows[1]["reason"] == "not larger than --max-size"


# --- The command ----------------------------------------------------------------------

runner = CliRunner()


class FakeProber:
    def __init__(self, answer: Any = good_probe) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def __call__(self, url: str) -> Probe:
        self.asked.append(url)
        return self.answer(url)

    def close(self) -> None:
        pass


@pytest.fixture
def index_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "config.yaml").write_text("index:\n  collection_key: COLL0001\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _command(
    monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]], annotated: set[str] | None = None
) -> tuple[FakeZotero, FakeFiles]:
    zot = seeded_zotero(items)
    files = FakeFiles({"PDFKEY01": PDF})
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr(cli, "build_client", lambda config: zot)
    monkeypatch.setattr(
        cli,
        "fetch_collections",
        lambda z: [{"key": "COLL0001", "data": {"key": "COLL0001", "name": "Papers"}}],
    )
    monkeypatch.setattr(cli, "fetch_collection_items", lambda z, key: items)
    monkeypatch.setattr(cli, "fetch_annotated_attachments", lambda z: annotated or set())
    monkeypatch.setattr(cli.FileDownloader, "from_config", lambda config: files)
    monkeypatch.setattr(cli, "LinkProber", FakeProber)
    monkeypatch.setenv("COLUMNS", "200")
    return zot, files


def test_dry_run_checks_and_moves_nothing(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zot, _files = _command(monkeypatch, library())

    result = runner.invoke(
        cli.app, ["shrink-storage", "--max-size", "1KB", "--dry-run", "--report", "out.json"]
    )

    assert result.exit_code == 0, result.output
    assert "Files that can move to the Zotero trash" in result.output
    assert "Dry run: nothing was moved" in result.output
    assert zot.patches == []
    rows = json.loads((index_folder / "out.json").read_text())
    assert rows[0]["section"] == "would move to trash"


def test_a_real_run_trashes_after_confirmation(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zot, _files = _command(monkeypatch, library())

    result = runner.invoke(
        cli.app, ["shrink-storage", "--max-size", "1KB", "--report", "out.csv"], input="y\n"
    )

    assert result.exit_code == 0, result.output
    assert "Move 1 file (4.9 KB) to the Zotero trash?" in result.output
    assert "Moved 1 file (4.9 KB) to the Zotero trash." in result.output
    assert "only when the trash is emptied" in result.output
    assert zot.stored["PDFKEY01"]["deleted"] == 1
    assert (index_folder / "shrink-backup" / "PDFKEY01.pdf").read_bytes() == PDF
    with (index_folder / "out.csv").open() as handle:
        (row,) = csv.DictReader(handle)
    assert row["section"] == "moved to trash"
    assert row["backup"].endswith("PDFKEY01.pdf")


def test_declining_moves_nothing_and_says_so(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zot, _files = _command(monkeypatch, library())

    result = runner.invoke(cli.app, ["shrink-storage", "--max-size", "1KB"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "Nothing was moved to the trash." in result.output
    assert "declined at the prompt" in result.output
    assert zot.patches == []


def test_a_read_only_key_stops_before_any_download(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zot, files = _command(monkeypatch, library())
    zot.access["user"]["write"] = False

    result = runner.invoke(cli.app, ["shrink-storage", "--max-size", "1KB", "--yes"])

    assert result.exit_code == 1
    assert "write access" in result.output
    assert files.fetched == []


def test_a_bad_max_size_is_an_error(index_folder: Path) -> None:
    result = runner.invoke(cli.app, ["shrink-storage", "--max-size", "big"])

    assert result.exit_code == 1
    assert "--max-size" in result.output


def test_kept_files_are_grouped_by_reason(
    index_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _command(monkeypatch, library(), annotated={"PDFKEY01"})

    result = runner.invoke(cli.app, ["shrink-storage", "--max-size", "1KB", "--yes"])

    assert result.exit_code == 0, result.output
    assert "No file can be moved to the trash." in result.output
    assert "1  has annotations" in result.output
