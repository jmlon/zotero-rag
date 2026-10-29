"""Phase 0 spike: convert the sample, measure it, and score it against the baseline.

Needs anydoc and MarkItDown on top of the project environment, without adding them
to pyproject.toml:

    uv run --with firecrawl-anydoc==0.2.4 --with 'markitdown[pdf,docx]==0.1.9' \
        python spike/anydoc/run_spike.py

Writes to spike/anydoc/out/:
    <key>.<engine>.md   every conversion, for reading by eye
    <key>.docmodel.json the anydoc document model of a few PDFs (page info?)
    results.json        one record per (file, engine), with the quality metrics
    grades.csv          template: grade each anydoc output good/bad by eye
and prints a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from zotero_rag.documents import DocumentFormat
from zotero_rag.extraction import ExtractionError, Segment, extract_segments

HERE = Path(__file__).parent
SAMPLES = HERE / "samples"
OUT = HERE / "out"
CONVERT_ONE = HERE / "convert_one.py"

FORMATS = {
    ".pdf": DocumentFormat.PDF,
    ".html": DocumentFormat.HTML,
    ".docx": DocumentFormat.DOCX,
    ".odt": DocumentFormat.ODT,
    ".epub": DocumentFormat.EPUB,
}

# --- Phase 1b metrics, prototyped -------------------------------------------------

WORD = re.compile(r"[^\W\d_]{4,}")
MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
MD_SYNTAX = re.compile(r"[#*_`>|~]+")
HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
CID = re.compile(r"\(cid:\d+\)")
MIN_SEGMENT_WORDS = 50
MIN_BASELINE_WORDS = 200
SINGLE_LETTER_OK = {"a", "i", "o", "y", "e"}  # common one-letter words (en, es)
GLUED_LETTERS = 20  # longer "words" are almost always several words missing their spaces


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    text = HTML_COMMENT.sub(" ", text)
    text = MD_LINK.sub(r"\1", text)
    return MD_SYNTAX.sub(" ", text).casefold()


def words(text: str) -> set[str]:
    return set(WORD.findall(normalise(text)))


def recall(baseline: set[str], candidate: set[str], flat: str = "") -> float | None:
    """Share of baseline words present in the candidate.

    A baseline word found *inside* the candidate's text also counts: pypdf breaks
    words on old LaTeX PDFs ("alidate", "approac"), and those fragments are not
    content the Markdown lost. A candidate that lost its spaces passes this too,
    which is what `glued_words` is for.
    """
    if not baseline:
        return None
    return sum(1 for w in baseline if w in candidate or w in flat) / len(baseline)


def garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = text.count("�") + sum(len(m) for m in CID.findall(text))
    bad += sum(1 for ch in text if unicodedata.category(ch) in ("Cc", "Co") and ch not in "\n\r\t")
    return bad / len(text)


def letter_ratio(text: str) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    return sum(ch.isalpha() for ch in visible) / len(visible) if visible else 0.0


def glued_words(text: str) -> float:
    """Share of words so long they must be several words with the spaces lost."""
    tokens = re.findall(r"[^\W\d_]+", normalise(text))
    return sum(len(t) >= GLUED_LETTERS for t in tokens) / len(tokens) if tokens else 0.0


def broken_spacing(text: str) -> float:
    tokens = [t for t in normalise(text).split() if t.isalpha()]
    if not tokens:
        return 0.0
    singles = sum(1 for t in tokens if len(t) == 1 and t not in SINGLE_LETTER_OK)
    return singles / len(tokens)


def assess(markdown: str, baseline: list[Segment]) -> dict[str, Any]:
    md_words = words(markdown)
    flat = normalise(markdown)
    base_text = "\n".join(segment.text for segment in baseline)
    base_words = words(base_text)
    per_segment = [
        (segment.number, recall(seg_words, md_words, flat))
        for segment in baseline
        if len(seg_words := words(segment.text)) >= MIN_SEGMENT_WORDS
    ]
    worst = min(per_segment, key=lambda pair: pair[1] or 0.0) if per_segment else None
    return {
        "baseline_words": len(base_words),
        "md_words": len(md_words),
        "verifiable": len(base_words) >= MIN_BASELINE_WORDS,
        "recall": _round(recall(base_words, md_words, flat)),
        "worst_segment": worst[0] if worst else None,
        "worst_segment_recall": _round(worst[1]) if worst else None,
        "garbage_ratio": round(garbage_ratio(markdown), 5),
        "letter_ratio": round(letter_ratio(markdown), 3),
        "broken_spacing": round(broken_spacing(markdown), 3),
        "glued_words": round(glued_words(markdown), 3),
        "page_markers": len(re.findall(r"<!--\s*page\s+\d+\s*-->", markdown, re.I)),
        "headings": len(re.findall(r"^#{1,6} ", markdown, re.M)),
        "tables": len(re.findall(r"^\|.*\|\s*$\n^\|[\s:|-]+\|\s*$", markdown, re.M)),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


# --- Running -----------------------------------------------------------------------


def pdf_facts(path: Path) -> dict[str, Any]:
    """Page and image counts: images are what Markdown cannot carry."""
    try:
        reader = PdfReader(str(path))
        images = 0
        for page in reader.pages:
            resources = page.get("/Resources") or {}
            xobjects = resources.get("/XObject") or {}
            images += sum(
                1 for ref in xobjects.values() if ref.get_object().get("/Subtype") == "/Image"
            )
        return {"pages": len(reader.pages), "images": images, "encrypted": reader.is_encrypted}
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return {"pdf_error": f"{type(exc).__name__}: {exc}"[:200]}


def baseline_for(path: Path) -> tuple[list[Segment], float, str]:
    document_format = FORMATS.get(path.suffix)
    if document_format is None:
        return [], 0.0, "no baseline extractor for this format"
    started = time.perf_counter()
    try:
        segments = extract_segments(path, document_format)
    except ExtractionError as exc:
        return [], time.perf_counter() - started, str(exc)
    return segments, time.perf_counter() - started, ""


def convert(engine: str, path: Path, key: str, timeout: int) -> dict[str, Any]:
    suffix = ".json" if engine == "docmodel" else ".md"
    target = OUT / f"{key}.{engine}{suffix}"
    try:
        done = subprocess.run(
            [sys.executable, "-I", str(CONVERT_ONE), engine, str(path), str(target)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"engine": engine, "ok": False, "error": "Timeout", "seconds": timeout}
    lines = done.stdout.strip().splitlines()
    if done.returncode != 0 or not lines:
        return {
            "engine": engine,
            "ok": False,
            "error": f"exit {done.returncode}",  # a native crash shows up here
            "message": done.stderr.strip()[-300:],
        }
    result: dict[str, Any] = json.loads(lines[-1])
    result["output"] = target.name if result.get("ok") else ""
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=int, default=300, help="Seconds per conversion.")
    parser.add_argument("--markitdown", type=int, default=10, help="Files to compare with.")
    parser.add_argument("--pages", type=int, default=8, help="PDFs for the per-page fallback.")
    parser.add_argument("--docmodel", type=int, default=3, help="PDFs to dump the model of.")
    args = parser.parse_args()

    manifest = json.loads((SAMPLES / "manifest.json").read_text())
    OUT.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    budget: Counter[str] = Counter()

    for number, entry in enumerate(manifest, start=1):
        path = SAMPLES / entry["path"]
        print(f"[{number}/{len(manifest)}] {entry['path']} {entry['title'][:60]!r}", flush=True)
        facts = pdf_facts(path) if path.suffix == ".pdf" else {}
        baseline, baseline_seconds, baseline_error = baseline_for(path)

        engines = ["anydoc"]
        if path.suffix == ".pdf":
            for engine, limit in (
                ("markitdown", args.markitdown),
                ("anydoc-pages", args.pages),
                ("docmodel", args.docmodel),
            ):
                if budget[engine] < limit:
                    engines.append(engine)
                    budget[engine] += 1
        elif budget["markitdown"] < args.markitdown and path.suffix in (".docx", ".html"):
            engines.append("markitdown")
            budget["markitdown"] += 1

        for engine in engines:
            result = convert(engine, path, entry["key"], args.timeout)
            record = {**entry, **facts, **result}
            record["baseline_seconds"] = round(baseline_seconds, 3)
            record["baseline_error"] = baseline_error
            if result.get("ok") and engine != "docmodel":
                markdown = (OUT / result["output"]).read_text(encoding="utf-8")
                record.update(assess(markdown, baseline))
            records.append(record)
            status = "ok" if result.get("ok") else result.get("error")
            print(
                f"    {engine:13} {status:18} {result.get('seconds', 0):7.2f}s "
                f"{result.get('peak_rss_mb', 0):7.1f}MB recall={record.get('recall')}",
                flush=True,
            )

    (OUT / "results.json").write_text(json.dumps(records, indent=2, default=str))
    write_grades(records)
    summarise(records)


def write_grades(records: list[dict[str, Any]]) -> None:
    """A sheet for the by-eye pass that calibrates the quality thresholds (D4)."""
    target = OUT / "grades.csv"
    if target.exists():
        print(f"\n{target.name} exists; not overwriting your grades.")
        return
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "file", "title", "recall", "worst_page_recall", "grade", "notes"])
        for r in records:
            if r["engine"] == "anydoc":
                writer.writerow(
                    [
                        r["key"],
                        r.get("output") or r["path"],
                        r["title"][:80],
                        r.get("recall"),
                        r.get("worst_segment_recall"),
                        "",
                        "",
                    ]
                )


def summarise(records: list[dict[str, Any]]) -> None:
    print("\n=== Summary ===")
    by_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_engine[r["engine"]].append(r)

    for engine, rows in by_engine.items():
        ok = [r for r in rows if r.get("ok")]
        print(f"\n{engine}: {len(ok)}/{len(rows)} succeeded")
        failures = Counter(r.get("error") for r in rows if not r.get("ok"))
        if failures:
            print("  failures:", dict(failures))
        if engine == "docmodel" or not ok:
            continue
        for label, key in (("seconds", "seconds"), ("peak RSS MB", "peak_rss_mb")):
            values = sorted(r[key] for r in ok if r.get(key) is not None)
            if values:
                print(
                    f"  {label:12} median {statistics.median(values):8.2f}  max {values[-1]:8.2f}"
                )
        for metric in (
            "recall",
            "worst_segment_recall",
            "garbage_ratio",
            "letter_ratio",
            "broken_spacing",
            "glued_words",
        ):
            values = sorted(r[metric] for r in ok if r.get(metric) is not None)
            if values:
                print(
                    f"  {metric:21} min {values[0]:.3f}  median "
                    f"{statistics.median(values):.3f}  max {values[-1]:.3f}"
                )
        by_format = Counter(Path(r["path"]).suffix for r in ok)
        print("  formats:", dict(by_format))
        print("  with page markers:", sum(1 for r in ok if r.get("page_markers")))

    pdf_rows = [r for r in by_engine.get("anydoc", []) if r["path"].endswith(".pdf")]
    seconds = sum(r.get("seconds", 0) for r in pdf_rows)
    pages = sum(r.get("pages", 0) or 0 for r in pdf_rows)
    if pages:
        print(f"\nanydoc PDF throughput: {pages / seconds:.1f} pages/s over {pages} pages")
    print(f"\nOutputs and results.json in {OUT}")


if __name__ == "__main__":
    main()
