"""Phase 1b of PLAN-markdown: judging a transcript against its source's text.

Each test recreates one failure mode Phase 0 found on real papers, at a size
where the thresholds apply: lost pages, lost spaces, broken font encodings, and
pypdf's own broken words, which must not count against a good transcript.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from zotero_rag.config import Config, QualityThresholds, load_config
from zotero_rag.documents import LocatorKind
from zotero_rag.extraction import Segment
from zotero_rag.markdown import body
from zotero_rag.quality import QualityVerdict, assess

THRESHOLDS = QualityThresholds()

# 343 distinct six-letter pseudo-words: "kakaka", "kakalo", …
VOCABULARY = [
    "".join(parts)
    for parts in itertools.product(("ka", "lo", "mi", "ne", "ru", "si", "to"), repeat=3)
]


def pages(count: int = 6, per_page: int = 55) -> list[Segment]:
    """A source of `count` pages, each with its own distinct words."""
    return [
        Segment(
            number=number,
            text=" ".join(VOCABULARY[(number - 1) * per_page : number * per_page]),
            kind=LocatorKind.PAGE,
        )
        for number in range(1, count + 1)
    ]


def as_markdown(segments: list[Segment]) -> str:
    return "\n\n".join(
        f"<!-- page {segment.number} -->\n\n## Part {segment.number}\n\n{segment.text}"
        for segment in segments
    )


def test_a_faithful_transcript_passes() -> None:
    source = pages()

    report = assess(as_markdown(source), source, THRESHOLDS)

    assert report.verdict is QualityVerdict.PASS
    assert report.recall == 1.0
    assert report.reasons == []
    assert report.summary().startswith("pass recall=1.00 worst=1.00@")


def test_a_dropped_page_fails_and_is_named() -> None:
    source = pages()
    # Five of six pages is still 83% recall overall, so the per-page check is the
    # one that must notice.
    report = assess(as_markdown(source[:2] + source[3:]), source, THRESHOLDS)

    assert report.verdict is QualityVerdict.FAIL
    assert report.worst_segment == (3, 0.0)
    assert any(reason.startswith("page 3 recall 0.000") for reason in report.reasons)


def test_missing_half_the_text_fails_overall_recall() -> None:
    source = pages()

    report = assess(as_markdown(source[:3]), source, THRESHOLDS)

    assert report.verdict is QualityVerdict.FAIL
    assert report.recall == 0.5
    assert any(reason.startswith("recall 0.500") for reason in report.reasons)


def test_lost_spaces_fail_even_though_every_word_is_there() -> None:
    # Phase 0's FL9A5KC7: recall is perfect because each word occurs inside the
    # run-together text; only the glued-words metric sees the damage.
    source = pages()
    words = " ".join(segment.text for segment in source).split()
    glued = " ".join("".join(words[i : i + 4]) for i in range(0, len(words), 4))

    report = assess(glued, source, THRESHOLDS)

    assert report.verdict is QualityVerdict.FAIL
    assert report.recall == 1.0
    assert report.glued_words > 0.9
    assert [reason.split()[:2] for reason in report.reasons] == [["glued", "words"]]


def test_fragments_in_the_baseline_do_not_count_against_the_transcript() -> None:
    # Phase 0's 6NCSR3YU: pypdf read "lokami" as "loka mi"-style fragments; the
    # transcript is right, the baseline is broken.
    source = pages()
    broken = [
        Segment(number=s.number, text=" ".join(w[:-1] for w in s.text.split()), kind=s.kind)
        for s in source
    ]
    broken[0] = Segment(
        number=1, text=" ".join(w[1:] for w in source[0].text.split()), kind=LocatorKind.PAGE
    )

    report = assess(as_markdown(source), broken, THRESHOLDS)

    assert report.verdict is QualityVerdict.PASS
    assert report.recall == 1.0


def test_cid_codes_and_replacement_characters_fail() -> None:
    source = pages()
    damaged = as_markdown(source) + "\n\n" + "(cid:12)�" * 60

    report = assess(damaged, source, THRESHOLDS)

    assert report.verdict is QualityVerdict.FAIL
    assert any(reason.startswith("garbage characters") for reason in report.reasons)


def test_letters_spread_out_by_spaces_fail() -> None:
    source = pages()
    spaced = as_markdown(source) + "\n\n" + " ".join("t h e r e s u l t s" for _ in range(40))

    report = assess(spaced, source, THRESHOLDS)

    assert report.verdict is QualityVerdict.FAIL
    assert any(reason.startswith("broken spacing") for reason in report.reasons)


def test_symbol_soup_fails_the_letter_ratio() -> None:
    source = pages()
    soup = as_markdown(source) + "\n\n" + "%$#@!&*^ 1234567890 " * 300

    report = assess(soup, source, THRESHOLDS)

    assert report.verdict is QualityVerdict.FAIL
    assert any(reason.startswith("letter ratio") for reason in report.reasons)


def test_page_markers_do_not_lower_the_letter_ratio() -> None:
    source = pages(count=40, per_page=8)
    marked = as_markdown(source)
    unmarked = "\n".join(line for line in marked.splitlines() if not line.startswith("<!--"))

    with_markers = assess(marked, source, THRESHOLDS).letter_ratio
    without_markers = assess(unmarked, source, THRESHOLDS).letter_ratio

    assert with_markers == without_markers


def test_a_source_too_short_to_compare_is_unverified() -> None:
    source = pages(count=1, per_page=40)

    report = assess(as_markdown(source), source, THRESHOLDS)

    assert report.verdict is QualityVerdict.UNVERIFIED
    assert report.recall is None
    assert "too few to check against" in report.reasons[0]


def test_a_scan_with_no_baseline_is_unverified_not_passed() -> None:
    report = assess("# Read by OCR\n\nAll of it.", [], THRESHOLDS, ocr_pages=[1, 2])

    assert report.verdict is QualityVerdict.UNVERIFIED
    assert any("pages read by OCR" in reason for reason in report.reasons)


def test_pages_read_by_ocr_leave_a_good_transcript_unverified() -> None:
    source = pages()

    report = assess(as_markdown(source), source, THRESHOLDS, ocr_pages=[7])

    assert report.verdict is QualityVerdict.UNVERIFIED
    assert report.recall == 1.0


def test_scanned_pages_left_empty_fail() -> None:
    source = pages()

    report = assess(as_markdown(source), source, THRESHOLDS, needs_ocr=[7, 9])

    assert report.verdict is QualityVerdict.FAIL
    assert "scanned pages left without text: 7, 9" in report.reasons


def test_a_visible_defect_outranks_unverified() -> None:
    glued = "".join(VOCABULARY[:40])

    report = assess(glued, [], THRESHOLDS)

    assert report.verdict is QualityVerdict.FAIL
    assert len(report.reasons) == 2  # glued words, then the missing baseline


def test_short_segments_are_left_out_of_the_worst_page() -> None:
    source = pages()
    source.append(Segment(number=7, text="kakaka lololo", kind=LocatorKind.PAGE))

    report = assess(as_markdown(source[:-1]), source, THRESHOLDS)

    assert report.verdict is QualityVerdict.PASS


def test_sections_are_named_as_sections() -> None:
    source = [Segment(number=s.number, text=s.text, kind=LocatorKind.SECTION) for s in pages()]

    report = assess(as_markdown(source[1:]), source, THRESHOLDS)

    assert any(reason.startswith("section 1 recall") for reason in report.reasons)


def test_stricter_thresholds_fail_what_the_defaults_pass() -> None:
    source = pages()
    nearly = as_markdown(source).replace(VOCABULARY[0], "")

    assert assess(nearly, source, THRESHOLDS).verdict is QualityVerdict.PASS
    strict = THRESHOLDS.model_copy(update={"min_recall": 1.0})
    assert assess(nearly, source, strict).verdict is QualityVerdict.FAIL


def test_front_matter_is_stripped_before_judging() -> None:
    text = "---\ntitle: Kakaka Lololo\nauthors: Mimimi\n---\n\nBody text\n"

    assert body(text) == "Body text\n"
    assert body("No front matter\n") == "No front matter\n"
    assert body("---\nunterminated\n") == "---\nunterminated\n"


# --- Configuration ----------------------------------------------------------------


def test_packaged_thresholds_are_the_calibrated_ones(tmp_path: Path) -> None:
    packaged = load_config(tmp_path).markdown.quality

    assert packaged == QualityThresholds()
    assert packaged.min_recall == 0.85
    assert packaged.min_worst_segment_recall == 0.80
    assert packaged.max_glued_words == 0.02


def test_thresholds_can_be_overridden_in_config() -> None:
    config = Config.model_validate({"markdown": {"quality": {"min_recall": 0.95}}})

    assert config.markdown.quality.min_recall == 0.95
    assert config.markdown.quality.max_glued_words == 0.02


def test_a_threshold_outside_zero_to_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        QualityThresholds(min_recall=1.5)


def test_math_notation_is_not_broken_spacing() -> None:
    # Phase 6 finding: a knowledge-graph survey's (h, r, t) triples and its
    # complexity tables (O(d), O(r k)) read as runs of lone consonants.
    source = pages()
    notation = " ".join("h r t d d r r k k where the triple holds" for _ in range(60))

    report = assess(as_markdown(source) + "\n\n" + notation, source, THRESHOLDS)

    assert report.broken_spacing == 0.0


def test_isolated_letters_are_not_broken_spacing() -> None:
    source = pages()
    prose = " ".join("let x be the head and y the tail" for _ in range(60))

    report = assess(as_markdown(source) + "\n\n" + prose, source, THRESHOLDS)

    assert report.broken_spacing == 0.0
