"""Phase 0 spike: check the Zotero write calls `convert` and `shrink-storage` need.

Runs ONLY against the throwaway test account in test_account.txt (repo root,
git-ignored) and refuses to touch the library configured in .env.

    uv run python spike/anydoc/zotero_writes.py

Each check prints PASS/FAIL/INFO; the full record goes to out/zotero_writes.json.
Created items are left in the test library for inspection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import traceback
from pathlib import Path
from typing import Any

from pyzotero import zotero

from zotero_rag.config import load_env

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
SAMPLES = HERE / "samples"
WORK = HERE / "out" / "writes"
SAMPLE_PDF = "847BBVQB.pdf"  # the smallest sample (90 KB)
MD_TAG = "zotero-rag:markdown"

results: list[dict[str, Any]] = []


def record(check: str, status: str, detail: Any = "") -> None:
    results.append({"check": check, "status": status, "detail": detail})
    print(f"{status:5} {check}: {detail}")


def credentials() -> tuple[str, str]:
    text = (ROOT / "test_account.txt").read_text()
    library = re.search(r"User ID for API calls:\s*(\d+)", text)
    key = re.search(r"API KEY:\s*(\S+)", text)
    if not (library and key):
        raise SystemExit("test_account.txt: could not find the user ID and API key lines")
    load_env()
    if library.group(1) == os.environ.get("ZOTERO_LIBRARY_ID"):
        raise SystemExit("Refusing: test_account.txt points at the real library from .env")
    return library.group(1), key.group(1)


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - Zotero's checksum


def data(zot: zotero.Zotero, key: str) -> dict[str, Any]:
    return zot.item(key)["data"]


def main() -> None:
    library_id, api_key = credentials()
    zot = zotero.Zotero(library_id, "user", api_key)
    WORK.mkdir(parents=True, exist_ok=True)

    # 1. Write access is visible up front (R6).
    access = zot.key_info().get("access", {})
    record("key_info access", "INFO", access)
    user_access = access.get("user", {})
    record(
        "key has write + file access",
        "PASS" if user_access.get("write") and user_access.get("files") else "FAIL",
        {k: user_access.get(k) for k in ("library", "files", "write")},
    )

    # 2. A parent item to hang everything on.
    parent = zot.item_template("journalArticle")
    parent["title"] = "zotero-rag Phase 0 write test"
    created = zot.create_items([parent])
    parent_key = created["successful"]["0"]["key"]
    record("create parent item", "PASS", parent_key)

    # 3. Upload a stored PDF the way a user's attachment looks.
    shutil.copy(SAMPLES / SAMPLE_PDF, WORK / "paper.pdf")
    pdf = zot.item_template("attachment", "imported_file")
    pdf.update(title="PDF", filename="paper.pdf", contentType="application/pdf")
    uploaded = zot.upload_attachments([pdf], parentid=parent_key, basedir=str(WORK))
    pdf_key = uploaded_key(uploaded)
    pdf_data = data(zot, pdf_key)
    record(
        "upload PDF attachment",
        "PASS" if pdf_data.get("md5") == md5_of(WORK / "paper.pdf") else "FAIL",
        {"key": pdf_key, "md5_matches": pdf_data.get("md5") == md5_of(WORK / "paper.pdf")},
    )

    # 4. Create the Markdown attachment with title, type, tag and provenance note in one go.
    md_path = WORK / "paper.md"
    md_path.write_text("---\ntitle: test\n---\n\n<!-- page 1 -->\n\n# Version one\n")
    note = f"zotero-rag:markdown v1\nsource={pdf_key} source_md5={pdf_data.get('md5')}"
    md = zot.item_template("attachment", "imported_file")
    md.update(
        title="Markdown (zotero-rag)",
        filename="paper.md",
        contentType="text/markdown",
        tags=[{"tag": MD_TAG}],
        note=note,
    )
    uploaded = zot.upload_attachments([md], parentid=parent_key, basedir=str(WORK))
    md_key = uploaded_key(uploaded)
    md_data = data(zot, md_key)
    stored = {
        "contentType": md_data.get("contentType"),
        "filename": md_data.get("filename"),
        "tags": md_data.get("tags"),
        "note_kept": md_data.get("note"),
        "md5_matches": md_data.get("md5") == md5_of(md_path),
    }
    ok = (
        stored["contentType"] == "text/markdown"
        and stored["md5_matches"]
        and {"tag": MD_TAG} in (md_data.get("tags") or [])
        and "source=" in (md_data.get("note") or "")
    )
    record("create Markdown attachment with metadata", "PASS" if ok else "FAIL", stored)

    # 5. Download it back: pyzotero maps text/markdown to "plain".
    try:
        body = zot.file(md_key)
        same = (body if isinstance(body, bytes) else str(body).encode()) == md_path.read_bytes()
        record("download Markdown via pyzotero", "PASS" if same else "FAIL", type(body).__name__)
    except Exception as exc:  # noqa: BLE001
        record("download Markdown via pyzotero", "FAIL", f"{type(exc).__name__}: {exc}")

    # 6. Replace the file in place: same key, new md5 (regenerating a STALE MD).
    old_md5 = md_data["md5"]
    md_path.write_text("---\ntitle: test\n---\n\n<!-- page 1 -->\n\n# Version two\n")
    replaced = zot.upload_attachments(
        [{"key": md_key, "filename": "paper.md", "md5": old_md5}], basedir=str(WORK)
    )
    after = data(zot, md_key)
    record(
        "replace Markdown file in place",
        "PASS" if after.get("md5") == md5_of(md_path) and after["key"] == md_key else "FAIL",
        {
            "result": {k: len(v) for k, v in replaced.items()},
            "md5_updated": after.get("md5") == md5_of(md_path),
            "note_survived": after.get("note") == note,
            "tags_survived": after.get("tags"),
        },
    )

    # 7. A replace with a stale md5 must be refused, not silently overwrite (edited-MD case).
    try:
        md_path.write_text("# Version three\n")
        zot.upload_attachments(
            [{"key": md_key, "filename": "paper.md", "md5": old_md5}], basedir=str(WORK)
        )
        record("replace with stale md5 refused", "FAIL", "upload accepted")
    except Exception as exc:  # noqa: BLE001
        record("replace with stale md5 refused", "PASS", type(exc).__name__)

    # 8. A metadata update with a stale version is refused (412).
    stale = data(zot, md_key)
    stale["version"] = stale["version"] - 1
    stale["title"] = "should not be saved"
    try:
        zot.update_item(stale)
        record("update with stale version refused", "FAIL", "update accepted")
    except Exception as exc:  # noqa: BLE001
        record("update with stale version refused", "PASS", type(exc).__name__)

    # 9. An annotation on the PDF, then find it as a child of the PDF (R11).
    annotation = {
        "itemType": "annotation",
        "parentItem": pdf_key,
        "annotationType": "highlight",
        "annotationText": "spike highlight",
        "annotationComment": "",
        "annotationColor": "#ffd400",
        "annotationPageLabel": "1",
        "annotationSortIndex": "00000|000000|00000",
        "annotationPosition": json.dumps({"pageIndex": 0, "rects": [[100, 100, 200, 120]]}),
        "tags": [],
    }
    try:
        made = zot.create_items([annotation])
        record(
            "create annotation", "PASS" if made.get("successful") else "FAIL", made.get("failed")
        )
    except Exception as exc:  # noqa: BLE001
        record("create annotation", "FAIL", f"{type(exc).__name__}: {exc}")
    children = zot.children(pdf_key)
    kinds = [c["data"]["itemType"] for c in children]
    record("annotations listed as PDF children", "PASS" if "annotation" in kinds else "FAIL", kinds)
    record(
        "annotations in item listings",
        "INFO",
        {
            "parent children": [c["data"]["itemType"] for c in zot.children(parent_key)],
            "items(itemType=annotation)": len(zot.items(itemType="annotation")),
        },
    )

    # 10. A link attachment, the recovery path shrink requires.
    link = zot.item_template("attachment", "linked_url")
    link.update(
        title="Publisher PDF", url="https://arxiv.org/pdf/1706.03762", parentItem=parent_key
    )
    made = zot.create_items([link])
    record(
        "create linked_url attachment",
        "PASS" if made.get("successful") else "FAIL",
        made.get("failed"),
    )

    # 11. Move the PDF to the trash; how does it show up afterwards?
    item = data(zot, pdf_key)
    item["deleted"] = 1
    zot.update_item(item)
    trashed = data(zot, pdf_key)
    record(
        "move PDF to trash",
        "PASS" if trashed.get("deleted") else "FAIL",
        {
            "deleted flag": trashed.get("deleted"),
            "in /items/trash": any(i["key"] == pdf_key for i in zot.trash()),
            "in parent's children": any(c["key"] == pdf_key for c in zot.children(parent_key)),
            "in /items": any(i["key"] == pdf_key for i in zot.everything(zot.items())),
            "file still downloadable": _downloadable(zot, pdf_key),
            "annotation children": [c["data"]["itemType"] for c in zot.children(pdf_key)],
        },
    )

    record(
        "keys",
        "INFO",
        {"parent": parent_key, "pdf": pdf_key, "markdown": md_key},
    )
    print(
        "\nQuota: the Web API does not report storage use. Compare "
        "https://www.zotero.org/settings/storage now (PDF in trash) and after emptying the trash."
    )


def uploaded_key(result: dict[str, list[dict[str, Any]]]) -> str:
    """The new attachment's key. Zotero answers "unchanged" when the server already
    holds a file with this md5 (it deduplicates), which still attaches the file."""
    done = result["success"] + result["unchanged"]
    if not done:
        raise RuntimeError(f"upload failed: {result['failure']}")
    if result["unchanged"]:
        record("upload deduplicated by server", "INFO", done[0]["filename"])
    return str(done[0]["key"])


def _downloadable(zot: zotero.Zotero, key: str) -> bool:
    try:
        return bool(zot.file(key))
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - keep what was learned before the failure
        record("unexpected error", "FAIL", traceback.format_exc()[-800:])
    finally:
        (HERE / "out").mkdir(exist_ok=True)
        (HERE / "out" / "zotero_writes.json").write_text(json.dumps(results, indent=2))
