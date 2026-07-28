"""Text out of HTML, using only the standard library.

An HTML snapshot is a saved web page, so most of its markup is furniture:
navigation, scripts, cookie banners, share buttons. Extraction therefore drops
non-content elements outright and keeps the heading structure, because a heading
is the only locator a web page offers — a citation of `§ Methods` is useful in a
way that "character offset 8,400" is not.

`html.parser` rather than BeautifulSoup or lxml: §9 forbids heavyweight
dependencies, and a snapshot only needs text and headings.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

# Elements whose text is never content.
SKIP_ELEMENTS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "head",
        "nav",
        "footer",
        "aside",
        "form",
        "button",
        "svg",
        "template",
        "iframe",
    }
)
HEADING_ELEMENTS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
# Elements that end a line of text, so words either side do not run together.
BLOCK_ELEMENTS = HEADING_ELEMENTS | frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "td",
        "th",
        "section",
        "article",
        "blockquote",
        "pre",
        "figcaption",
        "dt",
        "dd",
        "hr",
    }
)

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class Section:
    """A heading and the text that follows it, until the next heading."""

    def __init__(self, title: str = "") -> None:
        self.title = title
        self.parts: list[str] = []

    @property
    def text(self) -> str:
        return normalise(" ".join(self.parts))


def normalise(text: str) -> str:
    """Collapse runs of spaces and blank lines, preserving paragraph breaks."""
    collapsed = _WHITESPACE.sub(" ", text)
    lines = [line.strip() for line in collapsed.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


class _SectionParser(HTMLParser):
    """Split a document into heading-delimited sections of plain text."""

    def __init__(self) -> None:
        # convert_charrefs (the default) turns &amp; and &#8212; into characters.
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = [Section()]
        self.title = ""
        self._skip_depth = 0
        self._in_heading = False
        self._in_title = False
        self._heading_parts: list[str] = []

    # --- element boundaries ---

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_ELEMENTS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag in HEADING_ELEMENTS and not self._skip_depth:
            self._in_heading = True
            self._heading_parts = []
        elif tag in BLOCK_ELEMENTS and not self._skip_depth:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_ELEMENTS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        elif tag in HEADING_ELEMENTS and self._in_heading:
            self._in_heading = False
            heading = normalise(" ".join(self._heading_parts))
            # A heading starts a new section, so its text is not attributed to the
            # section above it.
            self.sections.append(Section(title=heading))
            if heading:
                self._append(f"{heading}\n")
        elif tag in BLOCK_ELEMENTS and not self._skip_depth:
            self._append("\n")

    def handle_data(self, data: str) -> None:
        # The document title is checked first: `<title>` sits inside `<head>`, which
        # is skipped wholesale, yet the title is the only label a page without
        # headings can offer.
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        if self._in_heading:
            self._heading_parts.append(data)
            return
        self._append(data)

    def _append(self, text: str) -> None:
        self.sections[-1].parts.append(text)

    # `title` is the <head> title; headings live inside <body>.
    def error(self, message: str) -> None:  # pragma: no cover - stdlib hook, unused
        logger.debug("html parse issue: %s", message)


def parse_sections(html: str) -> tuple[list[Section], str]:
    """Split HTML into non-empty sections, plus the document `<title>`.

    A page with no headings yields a single untitled section holding all its text,
    which is exactly how a plain article should behave.
    """
    parser = _SectionParser()
    parser.feed(html)
    parser.close()
    sections = [section for section in parser.sections if section.text]
    return sections, normalise(parser.title)


def to_text(html: str) -> str:
    """All readable text of a document, headings included."""
    sections, _ = parse_sections(html)
    return "\n\n".join(section.text for section in sections)
