"""Phase 4 end-to-end: build a test collection in the throwaway account.

Writes ONLY to the account in test_account.txt (guarded), then writes an index
folder whose config.yaml is pinned to the new collection.

    uv run python spike/anydoc/e2e_setup.py <index-folder>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from pyzotero import zotero

sys.path.insert(0, str(Path(__file__).parent))
from zotero_writes import credentials  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"

# (title, [(sample file, content type, link mode)])
PAPERS = [
    ("E2E plain PDF", [("847BBVQB.pdf", "application/pdf", "imported_file")]),
    (
        "E2E PDF and snapshot",
        [
            ("H3ZU298G.pdf", "application/pdf", "imported_file"),
            ("RGB4MGUR.html", "text/html", "imported_url"),
        ],
    ),
    (
        "E2E PDF with the user's own Markdown",
        [("QJCA72WM.pdf", "application/pdf", "imported_file"), ("USER.md", "text/markdown", "")],
    ),
    ("E2E OpenDocument", [("92UUQMIJ.odt", "application/vnd.oasis.opendocument.text", "")]),
    (
        "E2E Word",
        [
            (
                "ISJ39HVV.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "",
            )
        ],
    ),
    ("E2E partly scanned report", [("E2Y8PQAZ.pdf", "application/pdf", "imported_file")]),
]


def main(folder: Path) -> None:
    library_id, api_key = credentials()
    zot = zotero.Zotero(library_id, "user", api_key)

    made = zot.create_collections([{"name": "zotero-rag e2e"}])
    collection_key = made["successful"]["0"]["key"]

    staging = folder / "upload"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "USER.md").write_text("# My own notes\n\nWritten by hand.\n")

    for title, files in PAPERS:
        paper = zot.item_template("journalArticle")
        paper.update(title=title, collections=[collection_key])
        parent = zot.create_items([paper])["successful"]["0"]["key"]
        for name, content_type, link_mode in files:
            if name != "USER.md":
                shutil.copy(SAMPLES / name, staging / name)
            template = zot.item_template("attachment", link_mode or "imported_file")
            template.update(title=name, filename=name, contentType=content_type)
            result = zot.upload_attachments([template], parentid=parent, basedir=str(staging))
            done = result["success"] + result["unchanged"]
            print(f"{title}: {name} -> {done[0]['key'] if done else result['failure']}")

    (folder / "config.yaml").write_text(
        f"index:\n  collection_key: {collection_key}\n  collection_name: zotero-rag e2e\n"
    )
    print(f"\ncollection {collection_key}; index folder {folder}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
