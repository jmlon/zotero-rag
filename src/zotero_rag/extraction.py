"""PDF text extraction.

Text is extracted **page by page** so the page number recorded in each chunk's
metadata is exact, which is what makes citations point somewhere useful.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from zotero_rag.errors import ZoteroRagError

logger = logging.getLogger(__name__)


class ExtractionError(ZoteroRagError):
    """The PDF could not be read at all."""


class Page(BaseModel):
    """One page's extracted text. `number` is 1-based, as printed in citations."""

    number: int
    text: str


def extract_pages(pdf_path: Path) -> list[Page]:
    """Extract non-empty pages from a PDF.

    A page whose text extraction fails is logged and skipped rather than aborting
    the document: one bad page should not cost the whole paper. A document that
    yields no text at all (typically a scan) returns an empty list, which the
    caller reports as skipped — there is no OCR in the PoC.
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

    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - extraction failures vary widely
            logger.warning("Page %d of %s could not be extracted: %s", number, pdf_path.name, exc)
            continue
        if text.strip():
            pages.append(Page(number=number, text=text))

    if not pages:
        logger.warning("No extractable text in %s (a scanned PDF?)", pdf_path.name)
    return pages
