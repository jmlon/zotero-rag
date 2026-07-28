"""The attachment file cache, including multipart-envelope repair.

`zot` here is a tiny stand-in exposing only `file`, the single pyzotero method
this module touches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_pdf_bytes
from zotero_rag.download import (
    AttachmentDownloadError,
    cache_path,
    ensure_cached,
    strip_multipart_envelope,
)

BOUNDARY = b"--WN4u7NJ1LnAVdiYAdOo-OsaubgISqVtw"


def wrap_in_multipart(pdf: bytes) -> bytes:
    """Reproduce the envelope seen on real files in Zotero storage."""
    header = (
        BOUNDARY + b"\r\n"
        b'Content-Disposition: form-data; name="file"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n"
    )
    return header + pdf + b"\r\n" + BOUNDARY + b"--\r\n"


class FakeZotero:
    def __init__(self, content: bytes | None = None) -> None:
        self.content = make_pdf_bytes(["Hello"]) if content is None else content
        self.calls: list[str] = []

    def file(self, itemkey: str, /) -> bytes:
        self.calls.append(itemkey)
        return self.content


def test_downloads_when_absent(tmp_path: Path) -> None:
    zot = FakeZotero()
    result = ensure_cached(zot, "ATT1", tmp_path / "cache")
    assert result == cache_path(tmp_path / "cache", "ATT1")
    assert result.read_bytes().startswith(b"%PDF")
    assert zot.calls == ["ATT1"]


def test_cache_directory_is_created(tmp_path: Path) -> None:
    cache = tmp_path / "nested" / "cache"
    ensure_cached(FakeZotero(), "ATT1", cache)
    assert cache.is_dir()


def test_existing_file_is_reused_without_downloading(tmp_path: Path) -> None:
    """The whole point of the cache: re-indexing must not re-fetch."""
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = make_pdf_bytes(["Cached content"])
    cache_path(cache, "ATT1").write_bytes(cached)

    zot = FakeZotero()
    result = ensure_cached(zot, "ATT1", cache)

    assert zot.calls == []
    assert result.read_bytes() == cached


def test_empty_cached_file_is_re_downloaded(tmp_path: Path) -> None:
    """A zero-byte leftover from an interrupted run must not be trusted."""
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_path(cache, "ATT1").write_bytes(b"")

    zot = FakeZotero()
    ensure_cached(zot, "ATT1", cache)
    assert zot.calls == ["ATT1"]


def test_download_producing_no_file_is_an_error(tmp_path: Path) -> None:
    """Unsynced attachments return nothing; that must be reported."""
    with pytest.raises(AttachmentDownloadError, match="no file"):
        ensure_cached(FakeZotero(content=b""), "ATT1", tmp_path / "cache")


def test_pyzotero_errors_are_wrapped(tmp_path: Path) -> None:
    from pyzotero.zotero_errors import ResourceNotFoundError

    class NotFoundZotero:
        def file(self, itemkey: str, /) -> bytes:
            raise ResourceNotFoundError("gone")

    with pytest.raises(AttachmentDownloadError, match="ATT1"):
        ensure_cached(NotFoundZotero(), "ATT1", tmp_path / "cache")


def test_cache_path_uses_the_attachment_key(tmp_path: Path) -> None:
    assert cache_path(tmp_path, "ABC123").name == "ABC123.pdf"


# --- multipart envelope handling -------------------------------------------


def test_plain_pdf_is_returned_untouched() -> None:
    pdf = make_pdf_bytes(["Plain"])
    assert strip_multipart_envelope(pdf) == pdf


def test_envelope_is_stripped_to_a_valid_pdf() -> None:
    pdf = make_pdf_bytes(["Wrapped content"])
    cleaned = strip_multipart_envelope(wrap_in_multipart(pdf))
    assert cleaned.startswith(b"%PDF")
    assert cleaned.rstrip().endswith(b"%%EOF")
    assert BOUNDARY not in cleaned


def test_stripped_pdf_still_extracts(tmp_path: Path) -> None:
    """The repaired bytes must be readable by the extraction stage."""
    from zotero_rag.extraction import extract_pdf

    path = tmp_path / "repaired.pdf"
    path.write_bytes(strip_multipart_envelope(wrap_in_multipart(make_pdf_bytes(["Real text"]))))
    pages = extract_pdf(path)
    assert len(pages) == 1
    assert "Real text" in pages[0].text


def test_downloaded_envelope_is_stripped_before_caching(tmp_path: Path) -> None:
    zot = FakeZotero(content=wrap_in_multipart(make_pdf_bytes(["Body"])))
    result = ensure_cached(zot, "ATT1", tmp_path / "cache")
    assert result.read_bytes().startswith(b"%PDF")


def test_previously_cached_wrapped_file_is_repaired_without_re_downloading(
    tmp_path: Path,
) -> None:
    """Caches written before this repair existed must heal on the next run."""
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_path(cache, "ATT1").write_bytes(wrap_in_multipart(make_pdf_bytes(["Old"])))

    zot = FakeZotero()
    result = ensure_cached(zot, "ATT1", cache)

    assert zot.calls == []  # no re-download; the bytes were already local
    assert result.read_bytes().startswith(b"%PDF")


def test_data_without_any_pdf_header_is_left_alone() -> None:
    """Not a PDF: pass it through so extraction reports a clear error."""
    junk = b"<html>404 not found</html>"
    assert strip_multipart_envelope(junk) == junk


def test_header_beyond_the_envelope_limit_is_not_hunted() -> None:
    """Guards against treating a huge non-PDF blob as a wrapped PDF."""
    data = b"x" * 5000 + b"%PDF-1.4 trailing"
    assert strip_multipart_envelope(data) == data
