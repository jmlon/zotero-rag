"""M5: reading the formats a Zotero library actually holds.

The fixtures build genuine files — real ZIP containers with real XML — because a
string-matching stand-in would not prove the extractors can open what Word,
LibreOffice, and an EPUB builder actually write.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.conftest import make_docx_bytes, make_epub_bytes, make_odt_bytes
from zotero_rag import html_text
from zotero_rag.documents import (
    DocumentFormat,
    LocatorKind,
    cache_suffix,
    format_for,
    locator_kind_for,
)
from zotero_rag.extraction import (
    ExtractionError,
    extract_note,
    extract_segments,
)

FileFactory = Callable[[str, bytes | str], Path]


# --- content type mapping -------------------------------------------------


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("application/pdf", DocumentFormat.PDF),
        ("text/html", DocumentFormat.HTML),
        ("application/xhtml+xml", DocumentFormat.HTML),
        ("text/plain", DocumentFormat.TEXT),
        ("text/markdown", DocumentFormat.MARKDOWN),
        ("text/csv", DocumentFormat.TEXT),
        ("application/vnd.oasis.opendocument.text", DocumentFormat.ODT),
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            DocumentFormat.DOCX,
        ),
        ("application/epub+zip", DocumentFormat.EPUB),
    ],
)
def test_known_content_types_map_to_formats(content_type: str, expected: DocumentFormat) -> None:
    assert format_for(content_type) is expected


@pytest.mark.parametrize("content_type", ["image/png", "video/mp4", "", None, "application/zip"])
def test_unreadable_content_types_map_to_nothing(content_type: str | None) -> None:
    assert format_for(content_type) is None


def test_charset_parameters_and_case_are_ignored() -> None:
    """Both appear in real libraries and neither changes the format."""
    assert format_for("text/html; charset=UTF-8") is DocumentFormat.HTML
    assert format_for("Application/PDF") is DocumentFormat.PDF


def test_pdfs_keep_the_dot_pdf_cache_suffix() -> None:
    """Caches written before M5 must still hit, so this name cannot change."""
    assert cache_suffix(DocumentFormat.PDF) == ".pdf"
    assert cache_suffix(DocumentFormat.HTML) == ".html"


def test_only_pdfs_are_located_by_page() -> None:
    assert locator_kind_for(DocumentFormat.PDF) is LocatorKind.PAGE
    assert locator_kind_for(DocumentFormat.HTML) is LocatorKind.SECTION
    assert locator_kind_for(DocumentFormat.NOTE) is LocatorKind.NOTE


# --- HTML -----------------------------------------------------------------


def test_html_is_split_into_sections_at_headings(file_factory: FileFactory) -> None:
    path = file_factory(
        "page.html",
        "<html><head><title>Ignored</title></head><body>"
        "<h1>Introduction</h1><p>First body text.</p>"
        "<h2>Methods</h2><p>Second body text.</p>"
        "</body></html>",
    )
    segments = extract_segments(path, DocumentFormat.HTML)

    assert [segment.label for segment in segments] == ["Introduction", "Methods"]
    assert [segment.number for segment in segments] == [1, 2]
    assert all(segment.kind is LocatorKind.SECTION for segment in segments)
    assert "First body text." in segments[0].text
    assert "Second body text." not in segments[0].text


def test_scripts_styles_and_navigation_are_dropped(file_factory: FileFactory) -> None:
    """A snapshot is a whole web page; most of its markup is furniture."""
    path = file_factory(
        "page.html",
        "<html><body><nav>Home About Login</nav>"
        "<script>var tracking = 1;</script><style>.a{color:red}</style>"
        "<p>The actual article text.</p>"
        "<footer>Copyright notice</footer></body></html>",
    )
    text = extract_segments(path, DocumentFormat.HTML)[0].text

    assert "The actual article text." in text
    for furniture in ("tracking", "color:red", "Home About Login", "Copyright notice"):
        assert furniture not in text


def test_entities_are_decoded_and_blocks_do_not_run_together(file_factory: FileFactory) -> None:
    path = file_factory(
        "page.html", "<html><body><p>Fog&nbsp;computing &amp; edge</p><p>Second</p></body></html>"
    )
    text = extract_segments(path, DocumentFormat.HTML)[0].text
    assert "&" in text and "&amp;" not in text
    # Without block boundaries this would read "edgeSecond".
    assert "edgeSecond" not in text


def test_a_page_without_headings_is_labelled_by_its_title(file_factory: FileFactory) -> None:
    path = file_factory(
        "page.html",
        "<html><head><title>What is Fog Computing?</title></head>"
        "<body><p>Just one flat article.</p></body></html>",
    )
    segments = extract_segments(path, DocumentFormat.HTML)
    assert len(segments) == 1
    assert segments[0].label == "What is Fog Computing?"


def test_an_empty_page_yields_no_segments(file_factory: FileFactory) -> None:
    path = file_factory("empty.html", "<html><body><script>only()</script></body></html>")
    assert extract_segments(path, DocumentFormat.HTML) == []


def test_undeclared_encodings_do_not_fail_the_document(file_factory: FileFactory) -> None:
    """Snapshots come from the open web and are routinely mislabelled."""
    path = file_factory("latin.html", b"<html><body><p>caf\xe9 latte</p></body></html>")
    text = extract_segments(path, DocumentFormat.HTML)[0].text
    assert "latte" in text


# --- plain text and Markdown ---------------------------------------------


def test_markdown_headings_become_sections(file_factory: FileFactory) -> None:
    path = file_factory("notes.md", "# Title\n\nIntro line.\n\n## Details\n\nMore text.\n")
    segments = extract_segments(path, DocumentFormat.TEXT)
    assert [segment.label for segment in segments] == ["Title", "Details"]
    assert "Intro line." in segments[0].text


def test_plain_text_is_one_section(file_factory: FileFactory) -> None:
    path = file_factory("notes.txt", "Just some notes.\nOn two lines.\n")
    segments = extract_segments(path, DocumentFormat.TEXT)
    assert len(segments) == 1
    assert "Just some notes." in segments[0].text
    assert segments[0].label == ""


def test_an_empty_text_file_yields_no_segments(file_factory: FileFactory) -> None:
    assert extract_segments(file_factory("blank.txt", "   \n\n"), DocumentFormat.TEXT) == []


# --- DOCX and ODT ---------------------------------------------------------


def test_docx_sections_split_on_heading_styles(file_factory: FileFactory) -> None:
    path = file_factory(
        "paper.docx",
        make_docx_bytes(
            [
                ("", "Front matter."),
                ("Heading1", "Results"),
                ("", "We found something."),
                ("Heading2", "Discussion"),
                ("", "It means something."),
            ]
        ),
    )
    segments = extract_segments(path, DocumentFormat.DOCX)

    assert [segment.label for segment in segments] == ["", "Results", "Discussion"]
    assert "Front matter." in segments[0].text
    assert "We found something." in segments[1].text
    assert all(segment.kind is LocatorKind.SECTION for segment in segments)


def test_odt_sections_split_on_heading_elements(file_factory: FileFactory) -> None:
    path = file_factory(
        "paper.odt",
        make_odt_bytes([("p", "Opening."), ("h", "Method"), ("p", "How it was done.")]),
    )
    segments = extract_segments(path, DocumentFormat.ODT)

    assert [segment.label for segment in segments] == ["", "Method"]
    assert "How it was done." in segments[1].text


def test_a_corrupt_container_is_an_extraction_error(file_factory: FileFactory) -> None:
    """One unreadable file must fail as a skipped attachment, not crash the run."""
    path = file_factory("broken.docx", b"this is not a zip file")
    with pytest.raises(ExtractionError):
        extract_segments(path, DocumentFormat.DOCX)


def test_a_zip_without_the_expected_body_is_an_extraction_error(
    file_factory: FileFactory,
) -> None:
    path = file_factory("empty.odt", make_docx_bytes([("", "wrong container layout")]))
    with pytest.raises(ExtractionError):
        extract_segments(path, DocumentFormat.ODT)


# --- EPUB -----------------------------------------------------------------


def test_epub_chapters_follow_the_spine_not_the_file_names(file_factory: FileFactory) -> None:
    """The spine is the reading order, and it need not match archive order."""
    path = file_factory(
        "book.epub",
        make_epub_bytes(
            [
                ("a-first.xhtml", "Chapter One", "<p>Opening chapter.</p>"),
                ("b-second.xhtml", "Chapter Two", "<p>Closing chapter.</p>"),
            ],
            spine_order=True,  # OPF lists them reversed
        ),
    )
    segments = extract_segments(path, DocumentFormat.EPUB)
    assert [segment.label for segment in segments] == ["Chapter Two", "Chapter One"]
    assert "Closing chapter." in segments[0].text


def test_epub_without_a_readable_spine_falls_back_to_sorted_names(
    file_factory: FileFactory,
) -> None:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("b.xhtml", "<html><title>Two</title><body><p>Second</p></body></html>")
        archive.writestr("a.xhtml", "<html><title>One</title><body><p>First</p></body></html>")
    path = file_factory("nospine.epub", buffer.getvalue())

    segments = extract_segments(path, DocumentFormat.EPUB)
    assert [segment.label for segment in segments] == ["One", "Two"]


# --- notes ----------------------------------------------------------------


def test_a_note_is_one_segment_of_plain_text() -> None:
    segments = extract_note("<p>Key insight: <b>cite this</b> paper.</p>")
    assert len(segments) == 1
    assert segments[0].kind is LocatorKind.NOTE
    assert "Key insight: cite this paper." in segments[0].text


def test_an_empty_note_yields_no_segments() -> None:
    assert extract_note("<p></p>") == []
    assert extract_note("") == []


# --- shared normalisation -------------------------------------------------


def test_whitespace_is_collapsed_without_losing_paragraph_breaks() -> None:
    assert html_text.normalise("a    b\n\n\n\nc") == "a b\n\nc"


def test_no_extractor_for_an_unknown_format_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError):
        extract_segments(tmp_path / "x", DocumentFormat.NOTE)
