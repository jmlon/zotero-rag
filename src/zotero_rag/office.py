"""DOCX, ODT and EPUB text, using only `zipfile` and `xml.etree`.

All three are ZIP containers holding XML, so none of them needs a parser
dependency: DOCX keeps its body in `word/document.xml`, ODT in `content.xml`, and
EPUB is a set of XHTML files listed by a spine. `python-docx` would pull in `lxml`
to read one document in the target library, which §9 does not justify.

Headings are the useful locator in all three, so each returns titled sections in
document order.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from zotero_rag.html_text import Section, normalise, parse_sections

logger = logging.getLogger(__name__)

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
CONTAINER_NS = "{urn:oasis:names:tc:opendocument:xmlns:container}"
OPF_NS = "{http://www.idpf.org/2007/opf}"

DOCX_BODY = "word/document.xml"
ODT_BODY = "content.xml"
EPUB_CONTAINER = "META-INF/container.xml"


class OfficeError(Exception):
    """The container could not be opened or held no readable body."""


def _read_member(path: Path, name: str) -> bytes:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.read(name)
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise OfficeError(f"{path.name}: cannot read {name}: {exc}") from exc


def _parse_xml(data: bytes, source: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise OfficeError(f"{source}: malformed XML: {exc}") from exc


def _element_text(element: ElementTree.Element) -> str:
    """All text under an element, in document order."""
    return normalise("".join(element.itertext()))


def docx_sections(path: Path) -> list[Section]:
    """Sections of a DOCX, split on paragraphs styled as headings.

    Word marks a heading with a paragraph style whose name begins with "Heading"
    (`Heading1`, `Heading 2`, …), which is more reliable than guessing from font
    size.
    """
    root = _parse_xml(_read_member(path, DOCX_BODY), path.name)
    sections = [Section()]

    for paragraph in root.iter(f"{WORD_NS}p"):
        text = normalise("".join(node.text or "" for node in paragraph.iter(f"{WORD_NS}t")))
        if not text:
            continue
        if _is_word_heading(paragraph):
            sections.append(Section(title=text))
        sections[-1].parts.append(text)

    return [section for section in sections if section.text]


def _is_word_heading(paragraph: ElementTree.Element) -> bool:
    for style in paragraph.iter(f"{WORD_NS}pStyle"):
        value = style.get(f"{WORD_NS}val", "")
        if value.lower().startswith("heading"):
            return True
    return False


def odt_sections(path: Path) -> list[Section]:
    """Sections of an ODT, split on `text:h` heading elements.

    ODT is explicit where DOCX is stylistic: a heading is its own element type, so
    no style-name matching is needed.
    """
    root = _parse_xml(_read_member(path, ODT_BODY), path.name)
    sections = [Section()]

    # Headings and paragraphs must be visited together to preserve order.
    for element in root.iter():
        if element.tag == f"{TEXT_NS}h":
            title = _element_text(element)
            if title:
                sections.append(Section(title=title))
                sections[-1].parts.append(title)
        elif element.tag == f"{TEXT_NS}p":
            text = _element_text(element)
            if text:
                sections[-1].parts.append(text)

    return [section for section in sections if section.text]


def epub_sections(path: Path) -> list[Section]:
    """One section per EPUB document, in spine order where the spine is readable.

    The spine is the book's reading order; falling back to sorted archive names
    keeps a malformed EPUB usable, since alphabetical order is usually close to
    reading order for generated files.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = _epub_reading_order(archive)
            sections: list[Section] = []
            for name in names:
                try:
                    html = archive.read(name).decode("utf-8", errors="replace")
                except (KeyError, OSError) as exc:
                    logger.warning("%s: skipping %s: %s", path.name, name, exc)
                    continue
                parts, title = parse_sections(html)
                text = "\n\n".join(part.text for part in parts)
                if text:
                    sections.append(_section_with(title or Path(name).stem, text))
    except (zipfile.BadZipFile, OSError) as exc:
        raise OfficeError(f"{path.name}: not a readable EPUB: {exc}") from exc

    return sections


def _section_with(title: str, text: str) -> Section:
    section = Section(title=title)
    section.parts.append(text)
    return section


def _epub_reading_order(archive: zipfile.ZipFile) -> list[str]:
    """Content document names in spine order, or sorted names as a fallback."""
    fallback = sorted(
        name
        for name in archive.namelist()
        if name.lower().endswith((".xhtml", ".html", ".htm")) and not name.startswith("META-INF/")
    )
    try:
        container = _parse_xml(archive.read(EPUB_CONTAINER), EPUB_CONTAINER)
        rootfile = container.find(f".//{CONTAINER_NS}rootfile")
        opf_name = rootfile.get("full-path", "") if rootfile is not None else ""
        if not opf_name:
            return fallback

        opf = _parse_xml(archive.read(opf_name), opf_name)
        base = str(Path(opf_name).parent) if "/" in opf_name else ""
        hrefs = {
            item.get("id", ""): item.get("href", "")
            for item in opf.iter(f"{OPF_NS}item")
            if item.get("href")
        }
        ordered: list[str] = []
        for itemref in opf.iter(f"{OPF_NS}itemref"):
            href = hrefs.get(itemref.get("idref", ""), "")
            if not href:
                continue
            name = f"{base}/{href}" if base else href
            if name in archive.namelist():
                ordered.append(name)
        return ordered or fallback
    except (OfficeError, KeyError, OSError) as exc:
        logger.debug("no readable EPUB spine, falling back to sorted names: %s", exc)
        return fallback
