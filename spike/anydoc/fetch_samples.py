"""Phase 0 spike: pick a stratified sample of library attachments and download it.

Read-only against Zotero. Run from the repository root (its .env holds the keys):

    uv run python spike/anydoc/fetch_samples.py

Writes:
    spike/anydoc/samples/<key>.<ext>    the files
    spike/anydoc/samples/manifest.json  what each file is
    spike/anydoc/samples/library.json   library-wide counts that size the feature
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from zotero_rag.config import load_config, load_env
from zotero_rag.documents import format_for
from zotero_rag.download import strip_multipart_envelope
from zotero_rag.planning import LINKED_URL_MODE, is_in_storage
from zotero_rag.zotero_api import build_client

HERE = Path(__file__).parent
SAMPLES = HERE / "samples"

# Formats only conversion would read: index skips them today.
EXTRA_TYPES = {
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/msword": ".doc",
    "application/rtf": ".rtf",
    "text/rtf": ".rtf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.oasis.opendocument.presentation": ".odp",
}
SUFFIXES = {"pdf": ".pdf", "html": ".html", "docx": ".docx", "odt": ".odt", "epub": ".epub"}
# How many of each kind to draw. PDFs dominate the library, so they dominate the sample.
QUOTAS = {"pdf": 24, "docx": 3, "odt": 2, "epub": 2, "html": 3, "extra": 4}
STORED_MODES = ("imported_file", "imported_url")
MARKDOWN_TYPES = {"text/markdown", "text/x-markdown"}


def kind_of(content_type: str) -> tuple[str, str] | None:
    """(sample kind, file suffix), or None for types the spike ignores."""
    base = content_type.split(";", 1)[0].strip().lower()
    if base in EXTRA_TYPES:
        return "extra", EXTRA_TYPES[base]
    document_format = format_for(base)
    if document_format is None or document_format.value not in SUFFIXES:
        return None
    return document_format.value, SUFFIXES[document_format.value]


def library_stats(attachments: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts that decide how much `convert` and `shrink-storage` matter."""
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for data in attachments:
        if data.get("parentItem"):
            by_parent[data["parentItem"]].append(data)

    stored = [d for d in attachments if d.get("linkMode") in STORED_MODES]
    with_link = sum(
        1
        for siblings in by_parent.values()
        if any(s.get("linkMode") == LINKED_URL_MODE for s in siblings)
        and any(s.get("linkMode") in STORED_MODES and is_in_storage(s) for s in siblings)
    )
    user_markdown = sum(
        1
        for d in attachments
        if (d.get("contentType") or "").lower() in MARKDOWN_TYPES
        or (d.get("filename") or "").lower().endswith((".md", ".markdown"))
    )
    return {
        "attachments": len(attachments),
        "link_modes": Counter(d.get("linkMode") for d in attachments).most_common(),
        "content_types_stored": Counter(
            (d.get("contentType") or "?") for d in stored if is_in_storage(d)
        ).most_common(),
        "stored_but_not_synced": sum(1 for d in stored if not is_in_storage(d)),
        "items_with_stored_file_and_link_attachment": with_link,
        "markdown_attachments": user_markdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    load_env()
    zot = build_client(load_config())
    print("Listing every attachment in the library (read-only)...")
    attachments = [entry["data"] for entry in zot.everything(zot.items(itemType="attachment"))]

    SAMPLES.mkdir(parents=True, exist_ok=True)
    stats = library_stats(attachments)
    (SAMPLES / "library.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for data in attachments:
        if data.get("linkMode") not in STORED_MODES or not is_in_storage(data):
            continue
        kind = kind_of(data.get("contentType") or "")
        if kind:
            pools[kind[0]].append(data)

    rng = random.Random(args.seed)
    chosen: list[dict[str, Any]] = []
    for kind, quota in QUOTAS.items():
        pool = sorted(pools.get(kind, []), key=lambda d: d["key"])
        chosen.extend(rng.sample(pool, min(quota, len(pool))))
        print(f"{kind}: {len(pool)} available, sampled {min(quota, len(pool))}")

    manifest = []
    for data in chosen:
        _kind, suffix = kind_of(data.get("contentType") or "")  # type: ignore[misc]
        target = SAMPLES / f"{data['key']}{suffix}"
        if not target.is_file():
            try:
                raw = zot.file(data["key"])
            except Exception as exc:  # noqa: BLE001 - a spike records and moves on
                print(f"  ! {data['key']}: {exc}")
                continue
            target.write_bytes(strip_multipart_envelope(raw) if suffix == ".pdf" else raw)
        manifest.append(
            {
                "key": data["key"],
                "parent": data.get("parentItem", ""),
                "title": data.get("title", ""),
                "filename": data.get("filename", ""),
                "content_type": data.get("contentType", ""),
                "md5": data.get("md5", ""),
                "path": target.name,
                "bytes": target.stat().st_size,
            }
        )

    # Zotero metadata carries no file size, so the size spread is only known now.
    sizes = sorted(m["bytes"] for m in manifest)
    print(f"Downloaded {len(manifest)} files; sizes {sizes[0]:,} .. {sizes[-1]:,} bytes")
    (SAMPLES / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
