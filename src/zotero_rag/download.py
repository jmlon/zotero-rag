"""Attachment file cache.

Files are downloaded once into `<index>/cache/<attachment-key><suffix>` and
reused, so re-indexing never re-fetches a file that is already on disk. The
suffix comes from the attachment's format, and PDFs keep `.pdf` so caches
written before M5 still hit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from zotero_rag.documents import DocumentFormat, cache_suffix
from zotero_rag.errors import ZoteroRagError
from zotero_rag.zotero_api import NETWORK_ERRORS

logger = logging.getLogger(__name__)

CACHE_DIRNAME = "cache"

PDF_MAGIC = b"%PDF"
PDF_EOF = b"%%EOF"
# How far into the file we look for the PDF header before giving up.
_MAX_ENVELOPE_BYTES = 4096


class AttachmentFetcher(Protocol):
    """The one pyzotero method this module needs.

    Narrowing the dependency to a protocol keeps the seam explicit and lets the
    tests supply a local stand-in instead of a whole Zotero client.
    """

    def file(self, itemkey: str, /) -> bytes: ...


class AttachmentDownloadError(ZoteroRagError):
    """The attachment file could not be retrieved from Zotero."""


def cache_path(
    cache_dir: Path,
    attachment_key: str,
    document_format: DocumentFormat = DocumentFormat.PDF,
) -> Path:
    return cache_dir / f"{attachment_key}{cache_suffix(document_format)}"


def strip_multipart_envelope(data: bytes) -> bytes:
    """Unwrap a PDF that was stored inside a multipart/form-data envelope.

    Some files in Zotero storage were uploaded with the upload envelope kept
    verbatim, so their bytes begin with a `--boundary` line and end with a
    closing boundary. Zotero still serves them as `application/pdf`. pypdf can
    sometimes recover such a file by rescanning, but not reliably, so the
    envelope is removed before anything is written to the cache.

    Data that already starts with `%PDF` is returned untouched, which makes this
    a no-op for well-formed files.
    """
    if data.startswith(PDF_MAGIC):
        return data

    start = data.find(PDF_MAGIC, 0, _MAX_ENVELOPE_BYTES)
    if start < 0:
        # Not a PDF at all, or the header is further in than any real envelope.
        return data

    end = data.rfind(PDF_EOF)
    stripped = data[start : end + len(PDF_EOF)] if end > start else data[start:]
    logger.info(
        "stripped a %d-byte wrapper from the downloaded file (it was not raw PDF)",
        len(data) - len(stripped),
    )
    return stripped


def _write_clean(path: Path, data: bytes, document_format: DocumentFormat) -> None:
    """Write a downloaded file, unwrapping a stray upload envelope for PDFs."""
    cleaned = strip_multipart_envelope(data) if document_format is DocumentFormat.PDF else data
    if not cleaned:
        raise AttachmentDownloadError(f"Downloaded file for {path.stem} is empty")
    path.write_bytes(cleaned)


def ensure_cached(
    zot: AttachmentFetcher,
    attachment_key: str,
    cache_dir: Path,
    document_format: DocumentFormat = DocumentFormat.PDF,
) -> Path:
    """Return the local path for an attachment, downloading it if necessary."""
    target = cache_path(cache_dir, attachment_key, document_format)

    if target.is_file() and target.stat().st_size > 0:
        if document_format is not DocumentFormat.PDF:
            logger.debug("cache hit for %s", attachment_key)
            return target
        existing = target.read_bytes()
        if existing.startswith(PDF_MAGIC):
            logger.debug("cache hit for %s", attachment_key)
            return target
        # A file cached before envelope-stripping existed: repair it in place
        # rather than re-downloading bytes we already have.
        logger.info("repairing previously cached file for %s", attachment_key)
        _write_clean(target, existing, document_format)
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("downloading attachment %s", attachment_key)
    try:
        data = zot.file(attachment_key)
    except NETWORK_ERRORS as exc:
        raise AttachmentDownloadError(
            f"Could not download attachment {attachment_key}: {exc}"
        ) from exc

    if not data:
        raise AttachmentDownloadError(
            f"Zotero returned no file for attachment {attachment_key} "
            "(it may not be synced to the server)"
        )

    try:
        _write_clean(target, data, document_format)
    except OSError as exc:
        raise AttachmentDownloadError(f"Could not write {target}: {exc}") from exc

    return target
