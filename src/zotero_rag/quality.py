"""Does a Markdown transcript contain its source's text? (PRD-markdown D4)

The question needs an independent reading of the same file, and the project has
one: the extractors `index` uses (`extraction.extract_segments`). Every metric
compares the transcript against that baseline, page by page for PDFs, so a
dropped page cannot hide inside a good average.

Only a PASS lets `shrink-storage` trash an original. Where there is no baseline
to compare against — a scan, or pages read by OCR — the verdict is UNVERIFIED,
never PASS: nothing independent vouches for that text.

What no metric here sees: figures, equations and table structure. The PDF's
image count is reported alongside, as information.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field

from zotero_rag.config import QualityThresholds
from zotero_rag.extraction import Segment

# Words long enough to be distinctive; short ones match anywhere by accident.
_WORD = re.compile(r"[^\W\d_]{4,}")
_LETTERS = re.compile(r"[^\W\d_]+")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_MD_SYNTAX = re.compile(r"[#*_`>|~]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_CID = re.compile(r"\(cid:\d+\)")
# A "word" this long is several words that lost their spaces.
_GLUED_LETTERS = 20
# Real one-letter words (English, Spanish); any other lone letter is debris.
_SINGLE_LETTER_WORDS = frozenset({"a", "i", "o", "y", "e"})


class QualityVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIED = "unverified"


class QualityReport(BaseModel):
    """The verdict, the measurements behind it, and why it is not a PASS."""

    verdict: QualityVerdict
    baseline_words: int
    recall: float | None = None
    # (page or section number, its recall), for the weakest one measured.
    worst_segment: tuple[int, float] | None = None
    glued_words: float
    garbage_ratio: float
    letter_ratio: float
    broken_spacing: float
    reasons: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        """One line for reports and provenance: `pass recall=0.96 worst=0.93@12`."""
        parts = [str(self.verdict)]
        if self.recall is not None:
            parts.append(f"recall={self.recall:.2f}")
        if self.worst_segment is not None:
            number, value = self.worst_segment
            parts.append(f"worst={value:.2f}@{number}")
        return " ".join(parts)


def assess(
    markdown: str,
    baseline: Sequence[Segment] | None,
    thresholds: QualityThresholds,
    *,
    needs_ocr: Sequence[int] = (),
    ocr_pages: Sequence[int] = (),
) -> QualityReport:
    """Judge a transcript body (no front matter) against the source's segments.

    `needs_ocr` and `ocr_pages` come from the conversion: scanned pages left
    empty fail the transcript, and pages read by OCR leave it unverified.
    `baseline=None` means the format has no independent reader (pptx, rtf…),
    which leaves the transcript unverified for that reason.

    A FAIL outranks UNVERIFIED: glued words or garbage are visible without any
    baseline, and naming them is more useful than "could not check".
    """
    flat = _normalise(markdown)
    md_words = set(_WORD.findall(flat))
    readable = baseline is not None
    baseline = baseline or []
    base_words = _words("\n".join(segment.text for segment in baseline))
    # A baseline word counts as found when it occurs inside the transcript too:
    # pypdf breaks words on old LaTeX PDFs ("alidate"), and those fragments are
    # not text the transcript lost. Lost *spaces* are what `glued_words` catches.
    found = {word for word in base_words if word in md_words or word in flat}

    fails: list[str] = []
    unverified: list[str] = []
    recall: float | None = None
    worst: tuple[int, float] | None = None

    if not readable:
        unverified.append("no independent reading of this format to check against")
    elif len(base_words) < thresholds.min_baseline_words:
        unverified.append(
            f"the source has {len(base_words)} distinct words without OCR, "
            f"too few to check against (minimum {thresholds.min_baseline_words})"
        )
    else:
        recall = len(found) / len(base_words)
        if recall < thresholds.min_recall:
            fails.append(f"recall {recall:.3f} < {thresholds.min_recall}")
        worst = _worst_segment(baseline, found, thresholds.min_segment_words)
        if worst is not None and worst[1] < thresholds.min_worst_segment_recall:
            kind = baseline[0].kind if baseline else "segment"
            fails.append(
                f"{kind} {worst[0]} recall {worst[1]:.3f} < {thresholds.min_worst_segment_recall}"
            )

    glued = _glued_words(flat)
    garbage = _garbage_ratio(markdown)
    letters = _letter_ratio(markdown)
    spacing = _broken_spacing(flat)
    for failed, message in (
        (glued > thresholds.max_glued_words, f"glued words {glued:.3f}"),
        (garbage > thresholds.max_garbage_ratio, f"garbage characters {garbage:.4f}"),
        (letters < thresholds.min_letter_ratio, f"letter ratio {letters:.2f}"),
        (spacing > thresholds.max_broken_spacing, f"broken spacing {spacing:.3f}"),
    ):
        if failed:
            fails.append(message)
    if needs_ocr:
        fails.append(f"scanned pages left without text: {_pages(needs_ocr)}")
    if ocr_pages:
        unverified.append(
            f"pages read by OCR, with no text to check them against: {_pages(ocr_pages)}"
        )

    if fails:
        verdict = QualityVerdict.FAIL
    elif unverified:
        verdict = QualityVerdict.UNVERIFIED
    else:
        verdict = QualityVerdict.PASS
    return QualityReport(
        verdict=verdict,
        baseline_words=len(base_words),
        recall=_round(recall),
        worst_segment=None if worst is None else (worst[0], round(worst[1], 3)),
        glued_words=round(glued, 4),
        garbage_ratio=round(garbage, 5),
        letter_ratio=round(letters, 3),
        broken_spacing=round(spacing, 4),
        reasons=fails + unverified,
    )


def _worst_segment(
    baseline: Sequence[Segment], found: set[str], min_words: int
) -> tuple[int, float] | None:
    worst: tuple[int, float] | None = None
    for segment in baseline:
        words = _words(segment.text)
        if len(words) < min_words:
            continue
        value = len(words & found) / len(words)
        if worst is None or value < worst[1]:
            worst = (segment.number, value)
    return worst


def _normalise(text: str) -> str:
    """Comparable text: NFKC, casefolded, line-break hyphens rejoined, no markup."""
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _HTML_COMMENT.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    return _MD_SYNTAX.sub(" ", text).casefold()


def _words(text: str) -> set[str]:
    return set(_WORD.findall(_normalise(text)))


def _glued_words(flat: str) -> float:
    tokens = _LETTERS.findall(flat)
    return sum(len(token) >= _GLUED_LETTERS for token in tokens) / len(tokens) if tokens else 0.0


def _garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = text.count("�") + sum(len(match) for match in _CID.findall(text))
    bad += sum(1 for ch in text if unicodedata.category(ch) in ("Cc", "Co") and ch not in "\n\r\t")
    return bad / len(text)


def _letter_ratio(text: str) -> float:
    visible = [ch for ch in _HTML_COMMENT.sub(" ", text) if not ch.isspace()]
    return sum(ch.isalpha() for ch in visible) / len(visible) if visible else 0.0


def _broken_spacing(flat: str) -> float:
    tokens = [token for token in flat.split() if token.isalpha()]
    if not tokens:
        return 0.0
    singles = sum(1 for token in tokens if len(token) == 1 and token not in _SINGLE_LETTER_WORDS)
    return singles / len(tokens)


def _pages(numbers: Sequence[int]) -> str:
    return ", ".join(str(number) for number in numbers)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)
