"""Text extraction, one **segment** at a time.

A segment is the smallest unit a citation can point at: a PDF page, an HTML or
Word section, an EPUB chapter, a note. Splitting per segment before chunking is
what keeps the recorded locator exact — the same reason the PoC extracted PDFs
page by page (PRD §7).

Every extractor returns segments in document order; a document that yields no text
returns an empty list, which the caller reports as skipped rather than failed.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from zotero_rag import html_text, office, pdf_text
from zotero_rag.documents import DocumentFormat, LocatorKind, locator_kind_for
from zotero_rag.errors import ZoteroRagError
from zotero_rag.markdown import body

logger = logging.getLogger(__name__)

_PAGE_MARKER = re.compile(r"<!--\s*page\s+(\d+)\s*-->", re.IGNORECASE)


class ExtractionError(ZoteroRagError):
    """The document could not be read at all."""


class Segment(BaseModel):
    """One locatable piece of a document's text.

    `number` is 1-based and counts whatever `kind` says: pages for a PDF, sections
    for a snapshot. `label` carries a heading when the format has one, so a citation
    can read `§ Methods` instead of `§ 3`.
    """

    number: int
    text: str
    kind: LocatorKind = LocatorKind.PAGE
    label: str = ""


def extract_segments(path: Path, document_format: DocumentFormat) -> list[Segment]:
    """Extract text from a cached attachment according to its format."""
    if document_format is DocumentFormat.PDF:
        return extract_pdf(path)
    if document_format is DocumentFormat.HTML:
        return extract_html(path)
    if document_format is DocumentFormat.TEXT:
        return extract_text(path)
    if document_format is DocumentFormat.MARKDOWN:
        return extract_markdown(path)
    if document_format is DocumentFormat.DOCX:
        return _from_sections(office.docx_sections, path, document_format)
    if document_format is DocumentFormat.ODT:
        return _from_sections(office.odt_sections, path, document_format)
    if document_format is DocumentFormat.EPUB:
        return _from_sections(office.epub_sections, path, document_format)
    raise ExtractionError(f"No extractor for {document_format} ({path.name})")


def extract_pdf(pdf_path: Path) -> list[Segment]:
    """Extract non-empty pages from a PDF.

    A page whose text extraction fails is logged and skipped rather than aborting
    the document: one bad page should not cost the whole paper. A document that
    yields no text at all (typically a scan) returns an empty list — there is no
    OCR yet, so the caller reports it as skipped.
    """
    try:
        reader = PdfReader(str(pdf_path))
    except (PyPdfError, OSError, ValueError) as exc:
        raise ExtractionError(f"Cannot read {pdf_path.name}: {exc}") from exc

    if reader.is_encrypted:
        # An empty user password is common and pypdf can often open those.
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - pypdf raises assorted types here
            raise ExtractionError(
                f"{pdf_path.name} is encrypted and cannot be read: {exc}"
            ) from exc

    pages: list[Segment] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = pdf_text.clean(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - extraction failures vary widely
            logger.warning("Page %d of %s could not be extracted: %s", number, pdf_path.name, exc)
            continue
        if text.strip():
            pages.append(Segment(number=number, text=text, kind=LocatorKind.PAGE))

    if not pages:
        logger.warning("No extractable text in %s (a scanned PDF?)", pdf_path.name)
    return pages


def extract_html(path: Path) -> list[Segment]:
    """Sections of an HTML snapshot, split on its headings."""
    sections, title = html_text.parse_sections(_read_text(path))
    segments = _to_segments(sections, LocatorKind.SECTION)
    # A page with no headings gets its <title> as the one section's label, which is
    # the only human-meaningful locator such a document has.
    if len(segments) == 1 and not segments[0].label and title:
        segments[0].label = title
    if not segments:
        logger.warning("No extractable text in %s", path.name)
    return segments


def extract_text(path: Path) -> list[Segment]:
    """Plain text, Markdown or CSV.

    Markdown headings give real locators, so they are honoured when present;
    otherwise the whole file is one section.
    """
    content = _read_text(path)
    sections = _markdown_sections(content)
    segments = _to_segments(sections, LocatorKind.SECTION)
    if not segments:
        logger.warning("No extractable text in %s", path.name)
    return segments


def extract_markdown(path: Path) -> list[Segment]:
    """A Markdown file: a transcript's pages, or any Markdown's sections.

    A transcript made by `convert` marks each PDF page (`<!-- page 4 -->`), and
    those become PAGE segments, so a citation of the transcript reads `p. 4` just
    as one of the PDF did. Without markers — a transcript of a Word file, or
    Markdown the user wrote — headings give the sections. Front matter is
    metadata about the file, not its text, and is left out either way.
    """
    content = body(_read_text(path))
    pages = _marked_pages(content)
    if pages is not None:
        segments = [
            Segment(number=number, text=text, kind=LocatorKind.PAGE)
            for number, text in pages
            if text.strip()
        ]
    else:
        segments = _to_segments(_markdown_sections(content), LocatorKind.SECTION)
    if not segments:
        logger.warning("No extractable text in %s", path.name)
    return segments


def _marked_pages(content: str) -> list[tuple[int, str]] | None:
    """(page number, text) for each `<!-- page N -->`, or None without markers."""
    markers = list(_PAGE_MARKER.finditer(content))
    if not markers:
        return None
    pages: list[tuple[int, str]] = []
    for marker, following in zip(markers, [*markers[1:], None], strict=True):
        end = following.start() if following else len(content)
        pages.append((int(marker.group(1)), content[marker.end() : end].strip()))
    return pages


def extract_note(note_html: str) -> list[Segment]:
    """A Zotero note, whose body is HTML held in the item payload itself.

    Notes need no download, so this takes the markup rather than a path.
    """
    text = html_text.to_text(note_html)
    if not text:
        return []
    return [Segment(number=1, text=text, kind=LocatorKind.NOTE)]


def _markdown_sections(content: str) -> list[html_text.Section]:
    """Split on ATX headings (`# Title`), keeping the heading with its body."""
    sections = [html_text.Section()]
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and stripped.lstrip("#").strip():
            sections.append(html_text.Section(title=stripped.lstrip("#").strip()))
        sections[-1].parts.append(line + "\n")
    return [section for section in sections if section.text]


def _to_segments(sections: list[html_text.Section], kind: LocatorKind) -> list[Segment]:
    return [
        Segment(number=number, text=section.text, kind=kind, label=section.title)
        for number, section in enumerate(sections, start=1)
        if section.text
    ]


def _from_sections(reader: object, path: Path, document_format: DocumentFormat) -> list[Segment]:
    """Run one of the `office` readers and label the result for its format."""
    try:
        sections = reader(path)  # type: ignore[operator]
    except office.OfficeError as exc:
        raise ExtractionError(str(exc)) from exc
    segments = _to_segments(sections, locator_kind_for(document_format))
    if not segments:
        logger.warning("No extractable text in %s", path.name)
    return segments


def _read_text(path: Path) -> str:
    """Read a text-ish file, tolerating whatever encoding it was saved with.

    Snapshots come from the open web and are routinely mislabelled, so decoding
    replaces bad bytes instead of failing an otherwise readable document.
    """
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError as exc:
        raise ExtractionError(f"Cannot read {path.name}: {exc}") from exc
