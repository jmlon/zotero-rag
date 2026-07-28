"""PDF text extraction and segment-wise chunking."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from zotero_rag.chunking import chunk_segments
from zotero_rag.extraction import ExtractionError, Segment, extract_pdf

PdfFactory = Callable[[str, list[str]], Path]


def test_pages_are_extracted_with_one_based_numbers(pdf_factory: PdfFactory) -> None:
    path = pdf_factory("two.pdf", ["Hello page one", "Second page here"])
    pages = extract_pdf(path)
    assert [page.number for page in pages] == [1, 2]
    assert "Hello page one" in pages[0].text
    assert "Second page here" in pages[1].text


def test_blank_pages_are_dropped_but_numbering_stays_true(pdf_factory: PdfFactory) -> None:
    """Page 2 is empty, so page 3's recorded number must still be 3."""
    path = pdf_factory("gap.pdf", ["First", "   ", "Third"])
    pages = extract_pdf(path)
    assert [page.number for page in pages] == [1, 3]
    assert "Third" in pages[1].text


def test_pdf_without_text_yields_no_pages(pdf_factory: PdfFactory) -> None:
    """Stands in for a scanned document: no text, no OCR, nothing to embed."""
    path = pdf_factory("scan.pdf", ["", "  "])
    assert extract_pdf(path) == []


def test_unreadable_file_raises_extraction_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a PDF at all")
    with pytest.raises(ExtractionError):
        extract_pdf(path)


def test_missing_file_raises_extraction_error(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError):
        extract_pdf(tmp_path / "absent.pdf")


def test_chunking_splits_each_page_separately() -> None:
    pages = [Segment(number=1, text="a" * 250), Segment(number=2, text="b" * 250)]
    chunks = chunk_segments(pages, chunk_size=100, chunk_overlap=0)
    assert {chunk.locator for chunk in chunks} == {1, 2}
    # Index restarts per page, so (page, index) stays unique.
    assert [c.index for c in chunks if c.locator == 1] == [0, 1, 2]
    assert [c.index for c in chunks if c.locator == 2] == [0, 1, 2]


def test_small_page_is_a_single_chunk() -> None:
    chunks = chunk_segments([Segment(number=7, text="short text")], 1000, 200)
    assert len(chunks) == 1
    assert chunks[0].locator == 7
    assert chunks[0].index == 0
    assert chunks[0].text == "short text"


def test_chunks_respect_the_configured_size() -> None:
    pages = [Segment(number=1, text=" ".join(f"word{n}" for n in range(400)))]
    chunks = chunk_segments(pages, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_overlap_repeats_content_between_neighbours() -> None:
    pages = [Segment(number=1, text=" ".join(f"w{n}" for n in range(200)))]
    no_overlap = chunk_segments(pages, chunk_size=100, chunk_overlap=0)
    overlapped = chunk_segments(pages, chunk_size=100, chunk_overlap=50)
    assert len(overlapped) > len(no_overlap)


def test_whitespace_only_pieces_are_dropped() -> None:
    chunks = chunk_segments([Segment(number=1, text="   \n\n   ")], 100, 0)
    assert chunks == []


def test_no_pages_yields_no_chunks() -> None:
    assert chunk_segments([], 1000, 200) == []
