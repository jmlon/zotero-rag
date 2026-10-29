"""Markdown transcripts of attachments (PLAN-markdown, Phase 1).

An LLM reads plain text best, so `convert` turns a cached attachment into
GitHub-Flavored Markdown: anydoc for office formats and PDFs, `html_text` for
snapshots, which anydoc does not read.

PDFs are converted **one page at a time** (decision D2). anydoc's document model
does not cover PDFs, so this is the only way to know where pages begin, and each
page is preceded by a `<!-- page N -->` marker that keeps `p. 4` citations
possible once the transcript is indexed. It also means a PDF with a few scanned
pages still converts — whole, anydoc refuses such a document outright.

anydoc is imported lazily, so the rest of the tool keeps working on a platform
where its wheel is missing; only conversion then fails, with a clear message.
"""

from __future__ import annotations

import functools
import io
import logging
import re
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PyPdfError
from pypdf.generic import DictionaryObject

from zotero_rag import __version__, html_text, pdf_text
from zotero_rag.documents import DocumentFormat
from zotero_rag.errors import ZoteroRagError

logger = logging.getLogger(__name__)

OcrMode = Literal["off", "hosted"]

# Formats anydoc reads, by the names it uses for them (`documents.conversion_format_for`).
ANYDOC_FORMATS = frozenset(
    {"pdf", "docx", "odt", "epub", "doc", "rtf", "ppt", "pptx", "xlsx", "ods", "odp"}
)
PAGE_MARKER = "<!-- page {number} -->"
# A refused page with at least this many words in its text layer is not a scan or
# a blank, so its text layer stands in for the transcript of that page.
FALLBACK_MIN_WORDS = 20

_BLANK_LINES = re.compile(r"\n{3,}")


class Described(Protocol):
    """The bibliographic facts a transcript's front matter records.

    Both `planning.PlannedAttachment` and `conversion.SourceAttachment` have them.
    """

    title: str
    authors: str
    year: str
    doi: str
    item_key: str
    attachment_key: str
    zotero_url: str


class ConversionError(ZoteroRagError):
    """The attachment could not be turned into Markdown."""


class NeedsOcr(ConversionError):
    """A PDF has no readable text at all: every page with content is a scan."""

    def __init__(self, message: str, pages: list[int]) -> None:
        super().__init__(message)
        self.pages = pages


class EncryptedDocument(ConversionError):
    """The document is password-protected."""


class UnsupportedDocument(ConversionError):
    """The format cannot be converted, or needs no conversion."""


class Converted(BaseModel):
    """A transcript and what is known about how it was made."""

    markdown: str
    converter: str
    # PDFs only.
    pages: int | None = None
    # Pages that came out with no text: blank pages, and scans left unread.
    empty_pages: list[int] = Field(default_factory=list)
    # The subset of `empty_pages` that holds images, so OCR might recover text.
    needs_ocr: list[int] = Field(default_factory=list)
    # Pages read by the hosted OCR service rather than locally.
    ocr_pages: list[int] = Field(default_factory=list)
    # Pages anydoc refused whose plain text layer was used instead.
    text_layer_pages: list[int] = Field(default_factory=list)
    # Images the PDF draws. Markdown keeps none of them; reported, never judged.
    images: int = 0
    warnings: list[str] = Field(default_factory=list)


def convert(
    path: Path,
    source_format: str,
    *,
    ocr: OcrMode = "off",
    api_key: str = "",
) -> Converted:
    """Convert a cached attachment to Markdown.

    `source_format` is a `DocumentFormat` value or one of the formats only
    conversion reads (`documents.conversion_format_for`). With `ocr="hosted"`, PDF
    pages that hold only images are sent, one page at a time, to Firecrawl Parse;
    nothing else leaves the machine. `api_key` is optional: the service also works
    keyless, at lower limits.
    """
    if source_format == DocumentFormat.HTML:
        return _convert_html(path)
    if source_format == DocumentFormat.PDF:
        return _convert_pdf(path, ocr, api_key)
    if source_format in ANYDOC_FORMATS:
        return _convert_whole(path, source_format)
    raise UnsupportedDocument(f"{path.name}: {source_format} is not converted to Markdown")


def transcript(
    converted: Converted,
    attachment: Described,
    *,
    source_md5: str,
    generated: date,
) -> str:
    """The file that gets attached: YAML front matter, then the Markdown.

    The front matter tells an LLM — or a person — what it is reading and where it
    came from, so a transcript stays useful once separated from Zotero. `generated`
    is passed in rather than read from the clock, which keeps this deterministic.
    """
    fields: dict[str, Any] = {
        "title": attachment.title,
        "authors": attachment.authors,
        "year": attachment.year,
        "doi": attachment.doi,
        "zotero_item": attachment.item_key,
        "zotero_url": attachment.zotero_url,
        "source_attachment": attachment.attachment_key,
        "source_md5": source_md5,
        "converter": converted.converter,
        "generated": generated.isoformat(),
    }
    if converted.pages is not None:
        fields["pages"] = converted.pages
    # Said up front, so nobody mistakes a missing page for a missing argument.
    if converted.empty_pages:
        fields["empty_pages"] = converted.empty_pages
    if converted.ocr_pages:
        fields["ocr_pages"] = converted.ocr_pages
    if converted.text_layer_pages:
        fields["text_layer_pages"] = converted.text_layer_pages
    header = yaml.safe_dump(fields, sort_keys=False, allow_unicode=True, width=1000)
    return f"---\n{header}---\n\n{converted.markdown}"


def body(text: str) -> str:
    """A transcript without its YAML front matter; other Markdown is unchanged.

    Applied before judging a transcript, so the title and authors in its own
    header cannot count as text it recovered from the source.
    """
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    return text if end < 0 else text[end + len("\n---\n") :].lstrip("\n")


def normalise(markdown: str) -> str:
    """Strip trailing spaces, collapse blank-line runs, end with one newline.

    The same input must give byte-identical output: Zotero's md5 of the uploaded
    file is how a later run tells an untouched transcript from an edited one.
    """
    lines = [line.rstrip() for line in markdown.replace("\r\n", "\n").split("\n")]
    text = _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip("\n")
    return f"{text}\n" if text else ""


#: Bumped whenever this module's output changes for the same input, so that
#: `convert --reconvert` can find and redo transcripts made before the change.
#: 2: a PDF page anydoc refuses falls back to its text layer instead of a hole.
REVISION = 2


@functools.cache
def converter_name(source_format: str) -> str:
    """What produced a transcript, recorded in its provenance.

    The library version alone is not enough: the same anydoc can be driven
    differently, and a transcript made the old way must look outdated.
    """
    if source_format == DocumentFormat.HTML:
        return f"zotero-rag {__version__} r{REVISION}"
    try:
        return f"anydoc {version('firecrawl-anydoc')} r{REVISION}"
    except PackageNotFoundError:
        return f"anydoc r{REVISION}"


# --- PDF ------------------------------------------------------------------------


def _convert_pdf(path: Path, ocr: OcrMode, api_key: str) -> Converted:
    anydoc = _anydoc()
    reader = _open_pdf(path)
    texts: dict[int, str] = {}
    scans: dict[int, bytes] = {}
    empty: list[int] = []
    text_layer: list[int] = []
    warnings: list[str] = []
    images = 0

    for number, page in enumerate(reader.pages, start=1):
        page_images = _count_images(page)
        images += max(page_images, 0)
        try:
            data = _single_page(page)
        except (PyPdfError, ValueError, KeyError) as exc:
            warnings.append(f"page {number}: could not be separated ({exc})")
            empty.append(number)
            continue
        try:
            text = anydoc.to_markdown_bytes(data, "pdf")
        except anydoc.ConvertError as exc:
            # anydoc refuses some pages that have text: vector figures can read to
            # it as a scan. Their own text layer, plain as it is, beats a hole.
            fallback = pdf_text.text_layer(page)
            if pdf_text.word_count(fallback) >= FALLBACK_MIN_WORDS:
                texts[number] = fallback
                text_layer.append(number)
                continue
            if isinstance(exc, anydoc.NeedsOcrError):
                # anydoc says the same of a blank page as of a scan; only a page
                # with images can hold text OCR would find, and only those may be
                # uploaded. An unreadable resource tree (-1) might hide a scan.
                if page_images != 0:
                    scans[number] = data
            else:
                warnings.append(f"page {number}: {type(exc).__name__}: {exc}")
            empty.append(number)
            continue
        if text.strip():
            texts[number] = text
        else:
            empty.append(number)

    ocr_pages: list[int] = []
    if ocr == "hosted":
        for number, data in scans.items():
            texts[number] = _hosted_ocr(anydoc, data, api_key, path, number)
            ocr_pages.append(number)
            empty.remove(number)

    needs_ocr = [number for number in scans if number not in ocr_pages]
    if not texts:
        if needs_ocr:
            raise NeedsOcr(f"{path.name} is a scan: no page has text without OCR", needs_ocr)
        raise ConversionError(f"{path.name} has no text on any page")

    body = "\n\n".join(
        f"{PAGE_MARKER.format(number=number)}\n\n{texts.get(number, '').strip()}"
        for number in range(1, len(reader.pages) + 1)
    )
    return Converted(
        markdown=normalise(body),
        converter=converter_name("pdf"),
        pages=len(reader.pages),
        empty_pages=sorted(empty),
        needs_ocr=needs_ocr,
        ocr_pages=ocr_pages,
        text_layer_pages=text_layer,
        images=images,
        warnings=warnings,
    )


def _open_pdf(path: Path) -> PdfReader:
    try:
        reader = PdfReader(str(path))
    except (PyPdfError, OSError, ValueError) as exc:
        raise ConversionError(f"Cannot read {path.name}: {exc}") from exc
    if reader.is_encrypted:
        # An empty user password is common, and such a file opens without asking.
        try:
            reader.decrypt("")
            _ = len(reader.pages)
        except Exception as exc:  # noqa: BLE001 - pypdf raises assorted types here
            raise EncryptedDocument(f"{path.name} is password-protected") from exc
    return reader


def _single_page(page: Any) -> bytes:
    """One page as a PDF of its own. Written unencrypted, whatever the source was."""
    writer = PdfWriter()
    writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _count_images(page: Any) -> int:
    """Images the page draws, directly or inside form XObjects; -1 if unreadable."""
    try:
        return _images_in(page.get("/Resources"), depth=0)
    except (PyPdfError, AttributeError, KeyError, ValueError):
        return -1


def _images_in(resources: Any, depth: int) -> int:
    if depth > 3 or resources is None:
        return 0
    resources = resources.get_object()
    if not isinstance(resources, DictionaryObject):
        return 0
    xobjects = resources.get("/XObject")
    if xobjects is None:
        return 0
    count = 0
    for reference in xobjects.get_object().values():
        xobject = reference.get_object()
        subtype = xobject.get("/Subtype")
        if subtype == "/Image":
            count += 1
        elif subtype == "/Form":
            count += _images_in(xobject.get("/Resources"), depth + 1)
    return count


def _hosted_ocr(anydoc: ModuleType, data: bytes, api_key: str, path: Path, number: int) -> str:
    """One scanned page through Firecrawl Parse.

    A failure here (no credits, rate limit, network) would repeat for every page,
    so it fails the document rather than leaving a transcript with silent holes.
    """
    try:
        text: str = anydoc.to_markdown_bytes(data, "pdf", ocr="hosted", api_key=api_key or None)
    except anydoc.ConvertError as exc:
        raise ConversionError(f"{path.name}, page {number}: {exc}") from exc
    return text


# --- Other formats --------------------------------------------------------------


def _convert_whole(path: Path, source_format: str) -> Converted:
    """An office document or EPUB, converted in one piece.

    The format is always named: a cached file's suffix says little, and a minimal
    but valid container may lack the signature anydoc would detect it by.
    """
    anydoc = _anydoc()
    try:
        markdown = anydoc.to_markdown_bytes(path.read_bytes(), source_format)
    except OSError as exc:
        raise ConversionError(f"Cannot read {path.name}: {exc}") from exc
    except anydoc.EncryptedError as exc:
        raise EncryptedDocument(f"{path.name} is password-protected") from exc
    except anydoc.UnsupportedError as exc:
        raise UnsupportedDocument(f"{path.name}: {exc}") from exc
    except anydoc.ConvertError as exc:
        raise ConversionError(f"{path.name}: {type(exc).__name__}: {exc}") from exc
    if not markdown.strip():
        raise ConversionError(f"{path.name} has no text")
    return Converted(markdown=normalise(markdown), converter=converter_name(source_format))


def _convert_html(path: Path) -> Converted:
    """A snapshot, rendered from the sections `index` already extracts.

    Each heading becomes `##`; the page `<title>`, when there is one, heads the
    document. Same reader as indexing, so a snapshot's transcript and its indexed
    text never disagree about what the page says.
    """
    try:
        raw = path.read_bytes().decode("utf-8", errors="replace")
    except OSError as exc:
        raise ConversionError(f"Cannot read {path.name}: {exc}") from exc
    sections, title = html_text.parse_sections(raw)
    if not sections:
        raise ConversionError(f"{path.name} has no text")
    parts = [f"# {title}"] if title else []
    for section in sections:
        if not section.title:
            parts.append(section.text)
            continue
        # A section's text begins with its own heading, which indexing wants in
        # the chunk; here the heading is rendered once, as Markdown.
        body = section.text.removeprefix(section.title).lstrip("\n")
        parts.append(f"## {section.title}\n\n{body}" if body else f"## {section.title}")
    return Converted(markdown=normalise("\n\n".join(parts)), converter=converter_name("html"))


def _anydoc() -> ModuleType:
    try:
        import anydoc
    except ImportError as exc:
        raise ConversionError(
            "Markdown conversion needs the firecrawl-anydoc package, which is not "
            f"installed or has no build for this platform ({exc})"
        ) from exc
    return anydoc
