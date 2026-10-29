"""Phase 2 of PLAN-markdown: writing to Zotero, and downloading any file type.

Writes run against a recording stand-in for the few pyzotero methods they use;
the behaviours it imitates (deduplicated uploads, 412 on a stale md5, trashed
items) were each observed on a real test library in Phase 0
(`spike/anydoc/zotero_writes.py`). Downloads run through httpx's MockTransport,
which can model the redirect to file storage on another host.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pyzotero.zotero_errors import PreConditionFailedError, RequestEntityTooLargeError

from zotero_rag.config import Config
from zotero_rag.documents import DocumentFormat
from zotero_rag.download import AttachmentDownloadError, ensure_cached
from zotero_rag.errors import ZoteroApiError, ZoteroConflictError, ZoteroWriteAccessError
from zotero_rag.zotero_api import (
    FileDownloader,
    create_markdown_attachment,
    fetch_annotated_attachments,
    file_md5,
    replace_attachment_file,
    require_write_access,
    trash_item,
    update_attachment_note,
)

# --- A recording stand-in for pyzotero --------------------------------------------


class FakeZotero:
    def __init__(self) -> None:
        self.stored: dict[str, dict[str, Any]] = {}
        self.uploads: list[dict[str, Any]] = []
        self.patches: list[dict[str, Any]] = []
        self.access: dict[str, Any] = {"user": {"library": True, "files": True, "write": True}}
        self.dedupe = False
        self.server_md5: str | None = None  # simulate the server holding another file
        self.fail_with: Exception | None = None
        self.request: Any = None
        self.version = 100

    def key_info(self) -> dict[str, Any]:
        return {"access": self.access}

    def item_template(self, itemtype: str, linkmode: str | None = None) -> dict[str, Any]:
        return {"itemType": itemtype, "linkMode": linkmode, "title": "", "tags": [], "note": ""}

    def upload_attachments(
        self, attachments: list[dict[str, Any]], parentid: str | None = None, basedir: str = ""
    ) -> dict[str, list[dict[str, Any]]]:
        if self.fail_with:
            raise self.fail_with
        entry = dict(attachments[0])
        path = Path(basedir) / entry["filename"]
        self.version += 1
        if "key" in entry:  # replacing an existing file
            if entry["md5"] != self.stored[entry["key"]]["md5"]:
                raise PreConditionFailedError("412")
            self.stored[entry["key"]].update(md5=self.server_md5 or file_md5(path))
            self.stored[entry["key"]]["version"] = self.version
            self.uploads.append(entry)
            return {"success": [entry], "failure": [], "unchanged": []}
        key = f"NEW{len(self.stored):05d}"
        entry.update(key=key, parentItem=parentid)
        self.stored[key] = {
            **entry,
            "md5": self.server_md5 or file_md5(path),
            "version": self.version,
        }
        self.uploads.append(entry)
        bucket = "unchanged" if self.dedupe else "success"
        return {"success": [], "failure": [], "unchanged": [], bucket: [entry]}

    def item(self, key: str) -> dict[str, Any]:
        return {"key": key, "version": self.stored[key]["version"], "data": self.stored[key]}

    def update_item(self, payload: dict[str, Any]) -> bool:
        if self.fail_with:
            raise self.fail_with
        if payload["version"] != self.stored[payload["key"]]["version"]:
            raise PreConditionFailedError("412")
        self.version += 1
        self.stored[payload["key"]].update(payload, version=self.version)
        self.patches.append(payload)
        self.request = SimpleNamespace(headers={"Last-Modified-Version": str(self.version)})
        return True

    def everything(self, query: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return query


@pytest.fixture
def zot() -> FakeZotero:
    return FakeZotero()


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    path = tmp_path / "paper.md"
    path.write_text("<!-- page 1 -->\n\nVersion one\n")
    return path


# --- Write access -----------------------------------------------------------------


def test_write_access_passes_for_a_full_key(zot: FakeZotero) -> None:
    require_write_access(zot, Config())  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", ["write", "files"])
def test_write_access_needs_write_and_file_rights(zot: FakeZotero, missing: str) -> None:
    zot.access["user"][missing] = False

    with pytest.raises(ZoteroWriteAccessError, match="write access"):
        require_write_access(zot, Config())  # type: ignore[arg-type]


def test_write_access_for_a_group_library(zot: FakeZotero) -> None:
    config = Config.model_validate({"global": {"zotero_library_type": "group"}})
    config.global_.zotero_library_id = "4242"
    zot.access = {"groups": {"4242": {"library": True, "write": True}}}

    require_write_access(zot, config)  # type: ignore[arg-type]

    zot.access = {"groups": {"all": {"library": True, "write": False}}}
    with pytest.raises(ZoteroWriteAccessError):
        require_write_access(zot, config)  # type: ignore[arg-type]


# --- Creating and replacing transcripts --------------------------------------------


def test_create_sends_all_metadata_in_the_upload(zot: FakeZotero, markdown_file: Path) -> None:
    uploaded = create_markdown_attachment(
        zot,  # type: ignore[arg-type]
        "PARENT01",
        markdown_file,
        title="Markdown transcript",
        note="zotero-rag:markdown v1\n…",
        tags=["zotero-rag:markdown"],
    )

    sent = zot.uploads[0]
    assert sent["parentItem"] == "PARENT01"
    assert sent["filename"] == "paper.md"
    assert sent["contentType"] == "text/markdown"
    assert sent["tags"] == [{"tag": "zotero-rag:markdown"}]
    assert sent["note"].startswith("zotero-rag:markdown v1")
    assert uploaded.md5 == file_md5(markdown_file)
    assert uploaded.key == sent["key"]


def test_a_deduplicated_upload_counts_as_uploaded(zot: FakeZotero, markdown_file: Path) -> None:
    zot.dedupe = True

    uploaded = create_markdown_attachment(
        zot,  # type: ignore[arg-type]
        "PARENT01",
        markdown_file,
        title="t",
        note="n",
        tags=[],
    )

    assert uploaded.key.startswith("NEW")


def test_an_upload_the_server_does_not_hold_fails_verification(
    zot: FakeZotero, markdown_file: Path
) -> None:
    zot.server_md5 = "0" * 32

    with pytest.raises(ZoteroApiError, match="does not hold the uploaded file"):
        create_markdown_attachment(
            zot,  # type: ignore[arg-type]
            "PARENT01",
            markdown_file,
            title="t",
            note="n",
            tags=[],
        )


def test_a_full_storage_quota_is_named(zot: FakeZotero, markdown_file: Path) -> None:
    zot.fail_with = RequestEntityTooLargeError("413")

    with pytest.raises(ZoteroApiError, match="storage quota is full"):
        create_markdown_attachment(
            zot,  # type: ignore[arg-type]
            "PARENT01",
            markdown_file,
            title="t",
            note="n",
            tags=[],
        )


def test_replace_keeps_the_key_and_updates_the_md5(zot: FakeZotero, markdown_file: Path) -> None:
    first = create_markdown_attachment(
        zot,  # type: ignore[arg-type]
        "PARENT01",
        markdown_file,
        title="t",
        note="n",
        tags=[],
    )
    markdown_file.write_text("<!-- page 1 -->\n\nVersion two\n")

    second = replace_attachment_file(
        zot,  # type: ignore[arg-type]
        first.key,
        markdown_file,
        expected_md5=first.md5,
    )

    assert second.key == first.key
    assert second.md5 == file_md5(markdown_file) != first.md5


def test_replacing_an_edited_transcript_is_a_conflict(zot: FakeZotero, markdown_file: Path) -> None:
    first = create_markdown_attachment(
        zot,  # type: ignore[arg-type]
        "PARENT01",
        markdown_file,
        title="t",
        note="n",
        tags=[],
    )
    zot.stored[first.key]["md5"] = "f" * 32  # someone edited it in Zotero

    with pytest.raises(ZoteroConflictError, match="changed in Zotero"):
        replace_attachment_file(
            zot,  # type: ignore[arg-type]
            first.key,
            markdown_file,
            expected_md5=first.md5,
        )


# --- Notes and the trash ------------------------------------------------------------


def test_note_update_returns_the_new_version(zot: FakeZotero, markdown_file: Path) -> None:
    made = create_markdown_attachment(
        zot,  # type: ignore[arg-type]
        "PARENT01",
        markdown_file,
        title="t",
        note="old",
        tags=[],
    )

    version = update_attachment_note(zot, made.key, made.version, "new")  # type: ignore[arg-type]

    assert zot.stored[made.key]["note"] == "new"
    assert version == zot.stored[made.key]["version"] > made.version


def test_a_stale_version_is_a_conflict(zot: FakeZotero, markdown_file: Path) -> None:
    made = create_markdown_attachment(
        zot,  # type: ignore[arg-type]
        "PARENT01",
        markdown_file,
        title="t",
        note="n",
        tags=[],
    )

    with pytest.raises(ZoteroConflictError):
        update_attachment_note(zot, made.key, made.version - 1, "new")  # type: ignore[arg-type]


def test_trash_sets_deleted_and_never_deletes(zot: FakeZotero, markdown_file: Path) -> None:
    made = create_markdown_attachment(
        zot,  # type: ignore[arg-type]
        "PARENT01",
        markdown_file,
        title="t",
        note="n",
        tags=[],
    )

    trash_item(zot, made.key, made.version)  # type: ignore[arg-type]

    assert zot.patches == [{"key": made.key, "version": made.version, "deleted": 1}]
    assert made.key in zot.stored


def test_without_a_version_header_the_version_is_re_read(
    zot: FakeZotero, markdown_file: Path
) -> None:
    made = create_markdown_attachment(
        zot,  # type: ignore[arg-type]
        "PARENT01",
        markdown_file,
        title="t",
        note="n",
        tags=[],
    )
    original = zot.update_item

    def no_header(payload: dict[str, Any]) -> bool:
        original(payload)
        zot.request = SimpleNamespace(headers={})
        return True

    zot.update_item = no_header  # type: ignore[method-assign]

    assert trash_item(zot, made.key, made.version) == zot.stored[made.key]["version"]  # type: ignore[arg-type]


def test_annotated_attachments_are_their_parents() -> None:
    listing = [
        {"data": {"itemType": "annotation", "parentItem": "PDF00001"}},
        {"data": {"itemType": "annotation", "parentItem": "PDF00001"}},
        {"data": {"itemType": "annotation", "parentItem": "PDF00002"}},
        {"data": {"itemType": "annotation"}},
    ]
    client = SimpleNamespace(items=lambda **kwargs: listing, everything=lambda query: query)

    assert fetch_annotated_attachments(client) == {"PDF00001", "PDF00002"}  # type: ignore[arg-type]


# --- Downloading any file type ------------------------------------------------------

API = "https://api.zotero.org/users/123/items/ODT00001/file"
STORAGE = "https://files.example-storage.com/abc?signature=xyz"
ODT_BYTES = b"PK\x03\x04 an OpenDocument container"


def downloader(handler: Any, sleeps: list[float] | None = None) -> FileDownloader:
    return FileDownloader(
        "user",
        "123",
        "secret-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


def via_storage(content: bytes, *, compressed: bool = False) -> tuple[Any, list[httpx.Request]]:
    """The Web API's real shape: a 302 to file storage on another host."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "api.zotero.org":
            headers = {"Location": STORAGE}
            if compressed:
                headers["Zotero-File-Compressed"] = "Yes"
            return httpx.Response(302, headers=headers)
        return httpx.Response(
            200,
            content=content,
            headers={"Content-Type": "application/vnd.oasis.opendocument.text"},
        )

    return handler, seen


def test_odt_downloads_as_bytes() -> None:
    # The bug this replaces: pyzotero parsed this content type as JSON.
    handler, _ = via_storage(ODT_BYTES)

    assert downloader(handler).file("ODT00001") == ODT_BYTES


def test_the_api_key_is_not_sent_to_file_storage() -> None:
    handler, seen = via_storage(ODT_BYTES)

    downloader(handler).file("ODT00001")

    api, storage = seen
    assert str(api.url) == API
    assert api.headers["Authorization"] == "Bearer secret-key"
    assert "Authorization" not in storage.headers
    assert "secret-key" not in str(storage.headers)


def test_a_compressed_snapshot_is_unzipped() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("page.html", "<html>snapshot</html>")
    handler, _ = via_storage(buffer.getvalue(), compressed=True)

    assert downloader(handler).file("ODT00001") == b"<html>snapshot</html>"


def test_a_compressed_file_that_is_not_a_zip_is_an_error() -> None:
    handler, _ = via_storage(b"not a zip", compressed=True)

    with pytest.raises(httpx.HTTPError):
        downloader(handler).file("ODT00001")


def test_rate_limiting_is_waited_out() -> None:
    sleeps: list[float] = []
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, content=ODT_BYTES),
        ]
    )

    result = downloader(lambda request: next(responses), sleeps).file("ODT00001")

    assert result == ODT_BYTES
    assert sleeps == [7.0]


def test_rate_limiting_gives_up_eventually() -> None:
    sleeps: list[float] = []

    with pytest.raises(httpx.HTTPStatusError):
        downloader(lambda request: httpx.Response(429), sleeps).file("ODT00001")

    assert len(sleeps) == 3


def test_a_missing_file_reaches_the_user_as_a_download_error(tmp_path: Path) -> None:
    files = downloader(lambda request: httpx.Response(404))

    with pytest.raises(AttachmentDownloadError, match="404"):
        ensure_cached(files, "ODT00001", tmp_path / "cache", DocumentFormat.ODT)


def test_the_cache_works_with_the_new_downloader(tmp_path: Path) -> None:
    handler, seen = via_storage(ODT_BYTES)
    files = downloader(handler)

    path = ensure_cached(files, "ODT00001", tmp_path / "cache", DocumentFormat.ODT)
    again = ensure_cached(files, "ODT00001", tmp_path / "cache", DocumentFormat.ODT)

    assert path == again
    assert path.read_bytes() == ODT_BYTES
    assert len(seen) == 2  # one redirect + one storage fetch, then the cache


def test_credentials_are_required_at_the_first_download_not_before() -> None:
    from zotero_rag.errors import MissingCredentialError

    files = FileDownloader.from_config(Config())  # fine: maybe nothing needs fetching

    with pytest.raises(MissingCredentialError):
        files.file("ODT00001")
