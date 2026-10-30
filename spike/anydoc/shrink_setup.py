"""Phase 5 end-to-end: a test collection for `shrink-storage` in the throwaway account.

Writes ONLY to the account in test_account.txt (guarded), then writes an index
folder whose config.yaml is pinned to the new collection. Each paper is built to
end in one of the command's outcomes:

    uv run python spike/anydoc/shrink_setup.py <index-folder>
    cd <index-folder>
    zotero-rag convert --yes
    zotero-rag shrink-storage --max-size 1MB --dry-run
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import httpx
from pyzotero import zotero

sys.path.insert(0, str(Path(__file__).parent))
from zotero_writes import credentials  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"
ARXIV = "https://arxiv.org/pdf/1706.03762v7"

# (title, stored file, link URL): the stored file is uploaded byte for byte.
PAPERS = [
    ("Shrink: same file at the link", "arxiv.pdf", ARXIV),
    ("Shrink: link to a landing page", "arxiv.pdf", "https://arxiv.org/abs/1706.03762"),
    ("Shrink: link to another version", "arxiv.pdf", "https://arxiv.org/pdf/1706.03762v1"),
    ("Shrink: small file", "847BBVQB.pdf", ARXIV),
]


def main(folder: Path) -> None:
    library_id, api_key = credentials()
    zot = zotero.Zotero(library_id, "user", api_key)

    staging = folder / "upload"
    staging.mkdir(parents=True, exist_ok=True)
    response = httpx.get(ARXIV, follow_redirects=True, timeout=60)
    response.raise_for_status()
    (staging / "arxiv.pdf").write_bytes(response.content)
    shutil.copy(SAMPLES / "847BBVQB.pdf", staging / "847BBVQB.pdf")

    made = zot.create_collections([{"name": "zotero-rag shrink e2e"}])
    collection_key = made["successful"]["0"]["key"]

    for title, name, url in PAPERS:
        paper = zot.item_template("journalArticle")
        paper.update(title=title, collections=[collection_key])
        parent = zot.create_items([paper])["successful"]["0"]["key"]

        template = zot.item_template("attachment", "imported_file")
        template.update(title="Full Text PDF", filename=name, contentType="application/pdf")
        result = zot.upload_attachments([template], parentid=parent, basedir=str(staging))
        done = result["success"] + result["unchanged"]

        link = zot.item_template("attachment", "linked_url")
        link.update(title="Link", url=url, parentItem=parent)
        link_key = zot.create_items([link])["successful"]["0"]["key"]
        print(f"{title}: file {done[0]['key'] if done else result['failure']}, link {link_key}")

    (folder / "config.yaml").write_text(
        f"index:\n  collection_key: {collection_key}\n  collection_name: zotero-rag shrink e2e\n"
    )
    print(f"\ncollection {collection_key}; index folder {folder}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
