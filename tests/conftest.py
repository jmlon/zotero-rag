"""Shared test helpers."""

from __future__ import annotations

import io
import zipfile
from html import escape
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def make_pdf_bytes(pages_text: list[str]) -> bytes:
    """Build a real PDF whose pages contain the given text.

    Hand-assembled rather than pulling in reportlab: the PoC forbids heavyweight
    dependencies, and these are genuine PDFs that pypdf parses like any other.
    """
    writer = PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=200, height=200)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode())
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Contents")] = writer._add_object(stream)  # noqa: SLF001
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}  # noqa: SLF001
        )
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def pdf_factory(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Write a PDF with the given page texts and return its path."""

    def factory(name: str, pages_text: list[str]) -> Path:
        path = tmp_path / name
        path.write_bytes(make_pdf_bytes(pages_text))
        return path

    return factory


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ODT_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
OPF_NS = "http://www.idpf.org/2007/opf"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"


def make_docx_bytes(blocks: list[tuple[str, str]]) -> bytes:
    """Build a minimal but real DOCX from (style, text) pairs.

    Only `word/document.xml` matters to the extractor, so the archive omits the
    relationship parts Word itself would write. A style of "Heading1" marks a
    heading, which is how Word actually records one.
    """
    paragraphs = []
    for style, text in blocks:
        style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        paragraphs.append(f"<w:p>{style_xml}<w:r><w:t>{escape(text)}</w:t></w:r></w:p>")
    document = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{WORD_NS}"><w:body>{"".join(paragraphs)}</w:body></w:document>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def make_odt_bytes(blocks: list[tuple[str, str]]) -> bytes:
    """Build a minimal but real ODT from (kind, text) pairs, kind in {h, p}."""
    body = "".join(f"<text:{kind}>{escape(text)}</text:{kind}>" for kind, text in blocks)
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
        f' xmlns:text="{ODT_TEXT_NS}"><office:body><office:text>{body}'
        f"</office:text></office:body></office:document-content>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("content.xml", content)
    return buffer.getvalue()


def make_epub_bytes(chapters: list[tuple[str, str, str]], spine_order: bool = True) -> bytes:
    """Build a minimal but real EPUB from (filename, title, body html) triples.

    With `spine_order`, the OPF lists the chapters in reverse filename order, so a
    test can prove the reader follows the spine rather than the archive listing.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            f'<?xml version="1.0"?><container xmlns="{CONTAINER_NS}" version="1.0">'
            f'<rootfiles><rootfile full-path="OEBPS/content.opf" '
            f'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        for name, title, body in chapters:
            archive.writestr(
                f"OEBPS/{name}",
                f"<html><head><title>{escape(title)}</title></head><body>{body}</body></html>",
            )
        listed = list(reversed(chapters)) if spine_order else list(chapters)
        manifest = "".join(
            f'<item id="c{index}" href="{name}" media-type="application/xhtml+xml"/>'
            for index, (name, _, _) in enumerate(listed)
        )
        spine = "".join(f'<itemref idref="c{index}"/>' for index in range(len(listed)))
        archive.writestr(
            "OEBPS/content.opf",
            f'<?xml version="1.0"?><package xmlns="{OPF_NS}" version="3.0">'
            f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>",
        )
    return buffer.getvalue()


@pytest.fixture
def file_factory(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Write arbitrary bytes or text into the temp directory and return the path."""

    def factory(name: str, content: bytes | str) -> Path:
        path = tmp_path / name
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)
        return path

    return factory


@pytest.fixture(autouse=True)
def _no_price_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """`eval` and `compare-embeddings` fetch a list price from OpenRouter; tests stay offline."""
    from zotero_rag.llm import ModelPrice

    monkeypatch.setattr(
        "zotero_rag.cli.fetch_price",
        lambda endpoint, model: ModelPrice(prompt=1.5e-07, completion=6e-07),
    )
