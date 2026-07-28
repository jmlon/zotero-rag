"""Which Zotero attachments we can read, and what locator their text carries.

A PDF has pages; an HTML snapshot, a Word file, and a Zotero note do not. So the
page number of PRD §7.2 generalises to a **locator**: an ordinal plus a kind, and
optionally a label such as a heading. That is what lets a citation say `p. 4` for
one source and `§ Introduction` for another without pretending either is a page.
"""

from __future__ import annotations

import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class DocumentFormat(StrEnum):
    """A family of attachment we know how to extract text from."""

    PDF = "pdf"
    HTML = "html"
    TEXT = "txt"
    DOCX = "docx"
    ODT = "odt"
    EPUB = "epub"
    NOTE = "note"


class LocatorKind(StrEnum):
    """What a chunk's locator ordinal counts."""

    PAGE = "page"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    CHUNK = "chunk"
    NOTE = "note"


# Zotero reports the content type it stored the attachment with; that is the only
# reliable signal, since imported files keep arbitrary names.
CONTENT_TYPES: dict[str, DocumentFormat] = {
    "application/pdf": DocumentFormat.PDF,
    "text/html": DocumentFormat.HTML,
    "application/xhtml+xml": DocumentFormat.HTML,
    "text/plain": DocumentFormat.TEXT,
    "text/markdown": DocumentFormat.TEXT,
    "text/x-markdown": DocumentFormat.TEXT,
    "text/csv": DocumentFormat.TEXT,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        DocumentFormat.DOCX
    ),
    "application/vnd.oasis.opendocument.text": DocumentFormat.ODT,
    "application/epub+zip": DocumentFormat.EPUB,
}

# Suffix used in the cache. PDFs keep `.pdf` so caches predating M5 still hit.
CACHE_SUFFIXES: dict[DocumentFormat, str] = {
    DocumentFormat.PDF: ".pdf",
    DocumentFormat.HTML: ".html",
    DocumentFormat.TEXT: ".txt",
    DocumentFormat.DOCX: ".docx",
    DocumentFormat.ODT: ".odt",
    DocumentFormat.EPUB: ".epub",
}

LOCATOR_KINDS: dict[DocumentFormat, LocatorKind] = {
    DocumentFormat.PDF: LocatorKind.PAGE,
    DocumentFormat.HTML: LocatorKind.SECTION,
    DocumentFormat.TEXT: LocatorKind.SECTION,
    DocumentFormat.DOCX: LocatorKind.SECTION,
    DocumentFormat.ODT: LocatorKind.SECTION,
    DocumentFormat.EPUB: LocatorKind.SECTION,
    DocumentFormat.NOTE: LocatorKind.NOTE,
}


def format_for(content_type: str | None) -> DocumentFormat | None:
    """The format for a Zotero `contentType`, or `None` if we cannot read it.

    The parameters after `;` (`text/html; charset=utf-8`) are ignored, and the
    comparison is case-insensitive, because both appear in real libraries.
    """
    if not content_type:
        return None
    base = content_type.split(";", 1)[0].strip().lower()
    return CONTENT_TYPES.get(base)


def cache_suffix(document_format: DocumentFormat) -> str:
    """File suffix for a cached attachment of this format."""
    return CACHE_SUFFIXES.get(document_format, ".bin")


def locator_kind_for(document_format: DocumentFormat) -> LocatorKind:
    return LOCATOR_KINDS.get(document_format, LocatorKind.CHUNK)


def format_names() -> list[str]:
    """Format names accepted in `attachment_preference`."""
    return [item.value for item in DocumentFormat]
