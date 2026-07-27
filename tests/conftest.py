"""Shared test helpers."""

from __future__ import annotations

import io
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
