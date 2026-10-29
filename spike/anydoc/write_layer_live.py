"""Phase 2 live check: the real write-layer functions against the real Web API.

- Downloads (read-only) from the library in .env: the ODT pyzotero could not
  fetch, and a snapshot, compared byte for byte with what pyzotero returns.
- Writes ONLY to the throwaway account in test_account.txt (guarded).

    uv run python spike/anydoc/write_layer_live.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from pyzotero import zotero

sys.path.insert(0, str(Path(__file__).parent))
from zotero_writes import WORK, credentials, record, results  # noqa: E402

from zotero_rag.config import Config, load_config, load_env  # noqa: E402
from zotero_rag.errors import ZoteroConflictError  # noqa: E402
from zotero_rag.provenance import MARKDOWN_TAG, Provenance, parse  # noqa: E402
from zotero_rag.zotero_api import (  # noqa: E402
    FileDownloader,
    build_client,
    create_markdown_attachment,
    fetch_annotated_attachments,
    file_md5,
    replace_attachment_file,
    require_write_access,
    trash_item,
    update_attachment_note,
)

ODT_KEY = "92UUQMIJ"  # Ejercicios1-201610.odt in the real library
SNAPSHOT_KEY = "RGB4MGUR"


def downloads() -> None:
    load_env()
    config = load_config()
    files = FileDownloader.from_config(config)
    odt = files.file(ODT_KEY)
    record("download ODT (pyzotero failed)", "PASS" if odt[:2] == b"PK" else "FAIL", len(odt))
    ours = files.file(SNAPSHOT_KEY)
    theirs = build_client(config).file(SNAPSHOT_KEY)
    record(
        "snapshot matches pyzotero byte for byte", "PASS" if ours == theirs else "FAIL", len(ours)
    )


def writes() -> None:
    library_id, api_key = credentials()
    zot = zotero.Zotero(library_id, "user", api_key)
    config = Config()
    require_write_access(zot, config)
    record("require_write_access", "PASS", "")

    parent = zot.item_template("journalArticle")
    parent["title"] = "zotero-rag Phase 2 live check"
    parent_key = zot.create_items([parent])["successful"]["0"]["key"]

    WORK.mkdir(parents=True, exist_ok=True)
    pdf_path = WORK / "phase2.pdf"
    pdf_path.write_bytes((Path(__file__).parent / "samples" / "847BBVQB.pdf").read_bytes())
    pdf = zot.item_template("attachment", "imported_file")
    pdf.update(title="PDF", filename=pdf_path.name, contentType="application/pdf")
    upload = zot.upload_attachments([pdf], parentid=parent_key, basedir=str(WORK))
    pdf_key = (upload["success"] + upload["unchanged"])[0]["key"]

    md_path = WORK / "phase2.md"
    md_path.write_text("---\ntitle: live\n---\n\n<!-- page 1 -->\n\nVersion one\n")
    provenance = Provenance(
        source=pdf_key,
        source_md5=file_md5(pdf_path),
        md5=file_md5(md_path),
        converter="anydoc 0.2.4",
        quality="pass recall=0.99 worst=0.98@12",
    )
    made = create_markdown_attachment(
        zot,
        parent_key,
        md_path,
        title="Markdown transcript",
        note=provenance.to_note(),
        tags=[MARKDOWN_TAG],
    )
    stored = zot.item(made.key)["data"]
    record(
        "create_markdown_attachment + provenance round trip",
        "PASS" if parse(stored["note"]) == provenance else "FAIL",
        {"key": made.key, "contentType": stored["contentType"], "tags": stored["tags"]},
    )

    md_path.write_text("---\ntitle: live\n---\n\n<!-- page 1 -->\n\nVersion two\n")
    replaced = replace_attachment_file(zot, made.key, md_path, expected_md5=made.md5)
    record("replace_attachment_file", "PASS" if replaced.key == made.key else "FAIL", replaced.md5)

    try:
        replace_attachment_file(zot, made.key, md_path, expected_md5=made.md5)
        record("stale md5 -> ZoteroConflictError", "FAIL", "accepted")
    except ZoteroConflictError as exc:
        record("stale md5 -> ZoteroConflictError", "PASS", str(exc))

    updated = provenance.model_copy(update={"md5": replaced.md5})
    version = update_attachment_note(zot, made.key, replaced.version, updated.to_note())
    record(
        "update_attachment_note returns the new version",
        "PASS" if version == zot.item(made.key)["version"] else "FAIL",
        version,
    )
    try:
        update_attachment_note(zot, made.key, replaced.version, "stale")
        record("stale version -> ZoteroConflictError", "FAIL", "accepted")
    except ZoteroConflictError:
        record("stale version -> ZoteroConflictError", "PASS", "")

    annotation = {
        "itemType": "annotation",
        "parentItem": pdf_key,
        "annotationType": "highlight",
        "annotationText": "live",
        "annotationComment": "",
        "annotationColor": "#ffd400",
        "annotationPageLabel": "1",
        "annotationSortIndex": "00000|000000|00000",
        "annotationPosition": json.dumps({"pageIndex": 0, "rects": [[1, 1, 2, 2]]}),
        "tags": [],
    }
    zot.create_items([annotation])
    annotated = fetch_annotated_attachments(zot)
    record("fetch_annotated_attachments", "PASS" if pdf_key in annotated else "FAIL", annotated)

    pdf_version = zot.item(pdf_key)["version"]
    trash_item(zot, pdf_key, pdf_version)
    record(
        "trash_item",
        "PASS" if zot.item(pdf_key)["data"].get("deleted") else "FAIL",
        {"parent": parent_key, "pdf": pdf_key, "markdown": made.key, "date": str(date.today())},
    )


if __name__ == "__main__":
    downloads()
    writes()
    failed = [r for r in results if r["status"] == "FAIL"]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
