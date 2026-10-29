"""Phase 1 of PLAN-markdown: turning cached attachments into Markdown transcripts.

Conversions run the real anydoc on genuine files built by the conftest helpers.
Only hosted OCR is faked, because it is a network call to a third party.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, get_args

import anydoc
import pytest
import yaml
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from tests.conftest import make_docx_bytes, make_epub_bytes, make_odt_bytes, make_pdf_bytes
from zotero_rag import markdown
from zotero_rag.config import load_config
from zotero_rag.documents import DocumentFormat, conversion_format_for
from zotero_rag.markdown import (
    ConversionError,
    Converted,
    EncryptedDocument,
    NeedsOcr,
    UnsupportedDocument,
    convert,
    normalise,
    transcript,
)
from zotero_rag.planning import PlannedAttachment

FileFactory = Callable[[str, bytes | str], Path]


def scanned_pdf_bytes(pages: list[str | None]) -> bytes:
    """A PDF whose `None` pages hold only an image, like a scanned page."""
    reader = PdfReader(io.BytesIO(make_pdf_bytes([text or "" for text in pages])))
    writer = PdfWriter()
    for page, text in zip(reader.pages, pages, strict=True):
        added = writer.add_page(page)
        if text is None:
            image = DecodedStreamObject()
            image.set_data(b"\x00")
            image.update(
                {
                    NameObject("/Type"): NameObject("/XObject"),
                    NameObject("/Subtype"): NameObject("/Image"),
                    NameObject("/Width"): NumberObject(1),
                    NameObject("/Height"): NumberObject(1),
                    NameObject("/ColorSpace"): NameObject("/DeviceGray"),
                    NameObject("/BitsPerComponent"): NumberObject(8),
                }
            )
            added[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/XObject"): DictionaryObject(
                        {NameObject("/Im1"): writer._add_object(image)}  # noqa: SLF001
                    )
                }
            )
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# --- PDF, page by page (D2) -----------------------------------------------------


def test_pdf_pages_are_marked_in_order(file_factory: FileFactory) -> None:
    path = file_factory("paper.pdf", make_pdf_bytes(["First page text", "Second page text"]))

    result = convert(path, DocumentFormat.PDF)

    assert result.pages == 2
    assert result.empty_pages == []
    first = result.markdown.index("<!-- page 1 -->")
    second = result.markdown.index("<!-- page 2 -->")
    assert first < result.markdown.index("First page text") < second
    assert second < result.markdown.index("Second page text")
    assert result.converter.startswith("anydoc ")


def test_blank_page_keeps_its_marker_and_is_not_an_ocr_candidate(
    file_factory: FileFactory,
) -> None:
    # anydoc reports a blank page as needing OCR; a page with no image cannot be
    # a scan, so it must never be uploaded for OCR.
    path = file_factory("paper.pdf", make_pdf_bytes(["Opening words", "", "Closing words"]))

    result = convert(path, DocumentFormat.PDF)

    assert result.empty_pages == [2]
    assert result.needs_ocr == []
    assert "<!-- page 2 -->\n\n<!-- page 3 -->" in result.markdown


def test_partially_scanned_pdf_still_converts(file_factory: FileFactory) -> None:
    path = file_factory("report.pdf", scanned_pdf_bytes([None, "The findings", None]))

    result = convert(path, DocumentFormat.PDF)

    assert "The findings" in result.markdown
    assert result.empty_pages == [1, 3]
    assert result.needs_ocr == [1, 3]
    assert result.ocr_pages == []


def test_fully_scanned_pdf_raises_needs_ocr_naming_the_pages(file_factory: FileFactory) -> None:
    path = file_factory("scan.pdf", scanned_pdf_bytes([None, None]))

    with pytest.raises(NeedsOcr) as caught:
        convert(path, DocumentFormat.PDF)

    assert caught.value.pages == [1, 2]


def test_pdf_with_no_text_and_no_images_is_an_error(file_factory: FileFactory) -> None:
    path = file_factory("blank.pdf", make_pdf_bytes(["", ""]))

    with pytest.raises(ConversionError) as caught:
        convert(path, DocumentFormat.PDF)

    assert not isinstance(caught.value, NeedsOcr)


def test_password_protected_pdf(file_factory: FileFactory) -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf_bytes(["secret"])))
    writer.encrypt("a-password")
    buffer = io.BytesIO()
    writer.write(buffer)
    path = file_factory("locked.pdf", buffer.getvalue())

    with pytest.raises(EncryptedDocument):
        convert(path, DocumentFormat.PDF)


def test_pdf_with_an_empty_user_password_opens(file_factory: FileFactory) -> None:
    writer = PdfWriter(clone_from=io.BytesIO(make_pdf_bytes(["Readable anyway"])))
    writer.encrypt("", owner_password="owner")
    buffer = io.BytesIO()
    writer.write(buffer)
    path = file_factory("restricted.pdf", buffer.getvalue())

    assert "Readable anyway" in convert(path, DocumentFormat.PDF).markdown


def test_corrupt_pdf_is_a_conversion_error(file_factory: FileFactory) -> None:
    path = file_factory("broken.pdf", b"%PDF-1.4 this is not really a pdf")

    with pytest.raises(ConversionError):
        convert(path, DocumentFormat.PDF)


# --- Hosted OCR (D6) ------------------------------------------------------------


class FakeHostedAnydoc:
    """The real anydoc locally; records hosted calls instead of making them."""

    def __init__(self, reply: str | Exception = "Text the OCR service read") -> None:
        self.reply = reply
        self.hosted: list[dict[str, Any]] = []
        self.NeedsOcrError = anydoc.NeedsOcrError
        self.ConvertError = anydoc.ConvertError

    def to_markdown_bytes(self, data: bytes, fmt: str, **kwargs: Any) -> str:
        if kwargs.get("ocr") != "hosted":
            return anydoc.to_markdown_bytes(data, fmt)  # type: ignore[arg-type]
        self.hosted.append({"pages": len(PdfReader(io.BytesIO(data)).pages), **kwargs})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def fake_hosted(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeHostedAnydoc]:
    def install(reply: str | Exception = "Text the OCR service read") -> FakeHostedAnydoc:
        fake = FakeHostedAnydoc(reply)
        monkeypatch.setattr(markdown, "_anydoc", lambda: fake)
        return fake

    return install


def test_hosted_ocr_sends_only_scanned_pages_one_at_a_time(
    file_factory: FileFactory, fake_hosted: Callable[..., FakeHostedAnydoc]
) -> None:
    fake = fake_hosted()
    path = file_factory("report.pdf", scanned_pdf_bytes([None, "Local text", ""]))

    result = convert(path, DocumentFormat.PDF, ocr="hosted", api_key="key-123")

    assert fake.hosted == [{"pages": 1, "ocr": "hosted", "api_key": "key-123"}]
    assert result.ocr_pages == [1]
    assert result.empty_pages == [3]
    assert result.needs_ocr == []
    assert result.markdown.index("Text the OCR service read") < result.markdown.index("Local text")


def test_hosted_ocr_without_a_key_goes_keyless(
    file_factory: FileFactory, fake_hosted: Callable[..., FakeHostedAnydoc]
) -> None:
    fake = fake_hosted()
    path = file_factory("scan.pdf", scanned_pdf_bytes([None]))

    convert(path, DocumentFormat.PDF, ocr="hosted")

    assert fake.hosted[0]["api_key"] is None


def test_ocr_off_never_calls_the_service(
    file_factory: FileFactory, fake_hosted: Callable[..., FakeHostedAnydoc]
) -> None:
    fake = fake_hosted()
    path = file_factory("report.pdf", scanned_pdf_bytes([None, "Local text"]))

    convert(path, DocumentFormat.PDF)

    assert fake.hosted == []


def test_hosted_ocr_failure_fails_the_document(
    file_factory: FileFactory, fake_hosted: Callable[..., FakeHostedAnydoc]
) -> None:
    fake_hosted(anydoc.HostedError("Firecrawl Parse is out of credits"))
    path = file_factory("report.pdf", scanned_pdf_bytes([None, "Local text"]))

    with pytest.raises(ConversionError, match="out of credits"):
        convert(path, DocumentFormat.PDF, ocr="hosted")


# --- Other formats ----------------------------------------------------------------


def test_docx_headings_survive(file_factory: FileFactory) -> None:
    path = file_factory(
        "paper.docx",
        make_docx_bytes([("Heading1", "Results"), ("", "We found something.")]),
    )

    result = convert(path, DocumentFormat.DOCX)

    assert "Results" in result.markdown
    assert "We found something." in result.markdown
    assert result.pages is None


def test_odt_converts(file_factory: FileFactory) -> None:
    path = file_factory("notes.odt", make_odt_bytes([("h", "Method"), ("p", "How it was done.")]))

    result = convert(path, DocumentFormat.ODT)

    assert "# Method" in result.markdown
    assert "How it was done." in result.markdown


def test_epub_chapters_convert(file_factory: FileFactory) -> None:
    path = file_factory(
        "book.epub",
        make_epub_bytes([("c1.xhtml", "One", "<h1>Chapter One</h1><p>It began.</p>")]),
    )

    assert "# Chapter One" in convert(path, DocumentFormat.EPUB).markdown


def test_html_snapshot_uses_the_title_and_section_headings(file_factory: FileFactory) -> None:
    path = file_factory(
        "snapshot.html",
        "<html><head><title>An Article</title></head><body><nav>Menu</nav>"
        "<p>Lead paragraph.</p><h2>Methods</h2><p>What we did.</p></body></html>",
    )

    result = convert(path, DocumentFormat.HTML)

    assert result.markdown == "# An Article\n\nLead paragraph.\n\n## Methods\n\nWhat we did.\n"
    assert "Menu" not in result.markdown
    assert result.converter.startswith("zotero-rag ")


def test_html_with_no_text_is_an_error(file_factory: FileFactory) -> None:
    path = file_factory("empty.html", "<html><body><script>x()</script></body></html>")

    with pytest.raises(ConversionError):
        convert(path, DocumentFormat.HTML)


def test_text_is_not_converted(file_factory: FileFactory) -> None:
    path = file_factory("notes.txt", "Already text.")

    with pytest.raises(UnsupportedDocument):
        convert(path, DocumentFormat.TEXT)


def test_a_malformed_office_file_is_a_conversion_error(file_factory: FileFactory) -> None:
    path = file_factory("broken.docx", b"this is not a zip file")

    with pytest.raises(ConversionError):
        convert(path, DocumentFormat.DOCX)


def test_missing_anydoc_is_reported_not_raised_as_import_error(
    file_factory: FileFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "anydoc", None)
    path = file_factory("paper.pdf", make_pdf_bytes(["text"]))

    with pytest.raises(ConversionError, match="firecrawl-anydoc"):
        convert(path, DocumentFormat.PDF)


# --- Output shape -----------------------------------------------------------------


def test_normalise_is_deterministic_and_tidy() -> None:
    raw = "\n\n# Title  \r\n\n\n\nBody line\t\n\n\n"

    assert normalise(raw) == "# Title\n\nBody line\n"
    assert normalise(normalise(raw)) == normalise(raw)
    assert normalise("  \n\n") == ""


def test_transcript_front_matter_round_trips() -> None:
    attachment = PlannedAttachment(
        attachment_key="ATT1",
        item_key="ITEM1",
        title='Colons: "quotes" and — dashes',
        authors="Lovelace, A.; Babbage, C.",
        year="1843",
        doi="10.1000/xyz",
        collection_name="Computing",
        zotero_url="https://www.zotero.org/u/items/ITEM1",
    )
    converted = Converted(
        markdown="<!-- page 1 -->\n\nBody\n",
        converter="anydoc 0.2.4",
        pages=3,
        empty_pages=[2],
    )

    text = transcript(converted, attachment, source_md5="abc123", generated=date(2026, 10, 30))

    header, body = text.removeprefix("---\n").split("---\n\n", 1)
    fields = yaml.safe_load(header)
    assert fields["title"] == 'Colons: "quotes" and — dashes'
    assert fields["source_attachment"] == "ATT1"
    assert fields["source_md5"] == "abc123"
    assert fields["generated"] == "2026-10-30"
    assert fields["pages"] == 3
    assert fields["empty_pages"] == [2]
    assert "ocr_pages" not in fields
    assert body == "<!-- page 1 -->\n\nBody\n"


# --- Formats and configuration ----------------------------------------------------


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("application/pdf", "pdf"),
        ("text/html; charset=utf-8", "html"),
        ("application/vnd.oasis.opendocument.text", "odt"),
        ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx"),
        ("application/msword", "doc"),
        ("text/rtf", "rtf"),
        ("application/vnd.ms-excel", "xlsx"),
        ("text/plain", None),
        ("text/markdown", None),
        ("image/png", None),
        (None, None),
    ],
)
def test_conversion_format_for(content_type: str | None, expected: str | None) -> None:
    assert conversion_format_for(content_type) == expected


def test_conversion_only_formats_are_still_skipped_by_index() -> None:
    from zotero_rag.documents import format_for

    assert format_for("application/vnd.ms-powerpoint") is None


def test_firecrawl_key_is_optional_and_read_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert load_config(tmp_path).global_.firecrawl_api_key == ""

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    assert load_config(tmp_path).global_.firecrawl_api_key == "fc-test"


def test_every_anydoc_format_we_route_is_one_anydoc_knows() -> None:
    assert set(get_args(anydoc.Format)) >= markdown.ANYDOC_FORMATS


def test_images_are_counted_for_the_report(file_factory: FileFactory) -> None:
    path = file_factory("report.pdf", scanned_pdf_bytes([None, "Some text", None]))

    assert convert(path, DocumentFormat.PDF).images == 2
