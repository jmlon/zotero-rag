"""A PDF page's own text layer, as pypdf reads it, minus what is not prose.

Used twice: by `extraction` for indexing (and as the baseline the quality check
compares against), and by `markdown` as the fallback for a page anydoc refuses.

Figures drawn with LaTeXiT carry their LaTeX source as a base64 blob in the
page, and pypdf reads it out as "text" — hundreds of junk tokens per figure
that would be embedded, retrieved, and counted as words a transcript "missed".
"""

from __future__ import annotations

import re
from typing import Any

_LATEXIT = re.compile(r"<latexit\b.*?</latexit>", re.DOTALL)
# What is left of a blob whose tags pypdf split apart. A long run alone is not
# enough: text that lost its spaces ("AlgebraicMethodsforInteractive…") is long
# too, and must survive. Base64 always mixes in digits and `+` or `/`.
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/=]{60,}")
_DIGIT = re.compile(r"\d")
_BASE64_SYMBOL = re.compile(r"[+/]")
_WORD = re.compile(r"[^\W\d_]{2,}")


def clean(text: str) -> str:
    """Drop embedded LaTeX sources and base64 runs; keep everything else."""
    return _BASE64_RUN.sub(_drop_base64, _LATEXIT.sub(" ", text))


def _drop_base64(match: re.Match[str]) -> str:
    run = match.group(0)
    return " " if _DIGIT.search(run) and _BASE64_SYMBOL.search(run) else run


def text_layer(page: Any) -> str:
    """The page's cleaned text, or "" when pypdf cannot read it."""
    try:
        return clean(page.extract_text() or "")
    except Exception:  # noqa: BLE001 - pypdf raises assorted types on bad pages
        return ""


def word_count(text: str) -> int:
    return len(_WORD.findall(text))
