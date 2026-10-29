"""Convert one file with one engine and print a JSON result line.

Run as a subprocess by run_spike.py so a crash, a hang or a memory blow-up in one
document cannot take the run down, and so peak RSS is per document.

    python convert_one.py <engine> <input> <output.md>

Engines:
    anydoc        anydoc.to_markdown on the whole file
    anydoc-pages  PDF only: split with pypdf, convert each page, join with
                  <!-- page N --> markers (the D2 fallback)
    markitdown    MarkItDown, for comparison
    docmodel      dump the shape of anydoc.to_document() to look for page info
"""

from __future__ import annotations

import io
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any


def run_anydoc(source: Path) -> tuple[str, dict[str, Any]]:
    import anydoc

    return anydoc.to_markdown(str(source)), {"version": _version("firecrawl-anydoc")}


def run_anydoc_pages(source: Path) -> tuple[str, dict[str, Any]]:
    import anydoc
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(source))
    parts: list[str] = []
    failed: dict[int, str] = {}
    for number, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        buffer = io.BytesIO()
        writer.write(buffer)
        try:
            text = anydoc.to_markdown_bytes(buffer.getvalue(), "pdf")
        except anydoc.ConvertError as exc:
            failed[number] = type(exc).__name__
            text = ""
        parts.append(f"<!-- page {number} -->\n\n{text.strip()}\n")
    return "\n".join(parts), {"pages": len(reader.pages), "failed_pages": failed}


def run_markitdown(source: Path) -> tuple[str, dict[str, Any]]:
    from markitdown import MarkItDown

    result = MarkItDown(enable_plugins=False).convert(str(source))
    return result.text_content, {"version": _version("markitdown")}


def run_docmodel(source: Path) -> tuple[str, dict[str, Any]]:
    import anydoc

    document = anydoc.to_document(source.read_bytes())
    return json.dumps(_describe(document), indent=2, default=str), {}


def _describe(obj: Any, depth: int = 0, seen: set[int] | None = None) -> Any:
    """The attribute tree of an unknown object, a few levels deep and a few items wide."""
    seen = seen if seen is not None else set()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj if not isinstance(obj, str) else obj[:120]
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    if id(obj) in seen or depth > 5:
        return f"<{type(obj).__name__}>"
    seen.add(id(obj))
    if isinstance(obj, dict):
        return {str(k): _describe(v, depth + 1, seen) for k, v in list(obj.items())[:12]}
    if isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "len": len(obj),
            "first": [_describe(v, depth + 1, seen) for v in obj[:4]],
        }
    fields = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:  # noqa: BLE001 - introspection of an unknown type
            fields[name] = f"<error {type(exc).__name__}>"
            continue
        fields[name] = "<method>" if callable(value) else _describe(value, depth + 1, seen)
    return {"type": type(obj).__name__, "fields": fields}


def _version(distribution: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "?"


ENGINES = {
    "anydoc": run_anydoc,
    "anydoc-pages": run_anydoc_pages,
    "markitdown": run_markitdown,
    "docmodel": run_docmodel,
}


def main() -> None:
    engine, source, target = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    result: dict[str, Any] = {"engine": engine, "file": source.name}
    started = time.perf_counter()
    try:
        text, extra = ENGINES[engine](source)
        target.write_text(text, encoding="utf-8")
        result.update(ok=True, chars=len(text), **extra)
    except Exception as exc:  # noqa: BLE001 - every failure is a data point
        result.update(ok=False, error=type(exc).__name__, message=str(exc)[:300])
        for attribute in ("pages", "part", "limit"):  # NeedsOcrError.pages is a list
            if hasattr(exc, attribute):
                result[f"error_{attribute}"] = getattr(exc, attribute)
    result["seconds"] = round(time.perf_counter() - started, 3)
    # Linux reports ru_maxrss in KiB.
    result["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
