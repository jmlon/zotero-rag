"""Where a Markdown transcript came from, kept on its Zotero attachment.

The record lives in the attachment's `note` field, which every item listing
returns, so a run can tell whether a transcript is current without downloading
anything (PLAN-markdown §R1):

    zotero-rag:markdown v1
    source=ABCD2345
    source_md5=9e107d9d372bb6826bd81d3542a419d6
    md5=e4d909c290d0fb1ca068ffaddf22cbd0
    converter=anydoc 0.2.4
    quality=pass recall=0.96 worst=0.93@12

`md5` is the transcript as uploaded: when Zotero later reports another, a person
has edited it, and it is left alone. After `shrink-storage` trashes the source,
`source_removed` and `source_url` record that — the source then no longer shows
up in any listing, so this note is the only trace of it.

Parsing is strict on purpose. A note that does not parse is "not ours", and a
transcript that is not ours is never overwritten.
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel

from zotero_rag import html_text

HEADER = "zotero-rag:markdown v1"
#: Tagged as well, so the user can find every transcript in Zotero's tag selector.
MARKDOWN_TAG = "zotero-rag:markdown"

_KEY = re.compile(r"^[A-Z0-9]{8}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_LINE = re.compile(r"^([a-z][a-z0-9_]*)=(.*)$")


class Provenance(BaseModel):
    source: str
    source_md5: str
    md5: str
    converter: str
    # The source's format (`pdf`, `html`…). Absent from notes written before it
    # existed; `index` uses it to rank a transcript whose source is gone.
    source_format: str = ""
    quality: str = ""
    source_removed: date | None = None
    source_url: str = ""

    def to_note(self) -> str:
        lines = [
            HEADER,
            f"source={self.source}",
            f"source_md5={self.source_md5}",
            f"md5={self.md5}",
            f"converter={self.converter}",
        ]
        if self.source_format:
            lines.append(f"source_format={self.source_format}")
        if self.quality:
            lines.append(f"quality={self.quality}")
        if self.source_removed is not None:
            lines.append(f"source_removed={self.source_removed.isoformat()}")
        if self.source_url:
            lines.append(f"source_url={self.source_url}")
        return "\n".join(lines)


def parse(note: str | None) -> Provenance | None:
    """The provenance in an attachment note, or `None` if the note is not ours.

    Zotero keeps a note written through the API as plain text, but the desktop
    client turns it into HTML once anyone opens it, so markup is tolerated.
    Unknown keys are ignored, so a later version can add fields.
    """
    if not note:
        return None
    text = html_text.to_text(note) if "<" in note else note
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != HEADER:
        return None

    fields: dict[str, str] = {}
    for line in lines[1:]:
        match = _LINE.match(line)
        if match:
            fields.setdefault(match.group(1), match.group(2).strip())

    if not (
        _KEY.match(fields.get("source", ""))
        and _MD5.match(fields.get("source_md5", ""))
        and _MD5.match(fields.get("md5", ""))
        and fields.get("converter")
    ):
        return None
    removed: date | None = None
    if "source_removed" in fields:
        try:
            removed = date.fromisoformat(fields["source_removed"])
        except ValueError:
            return None
    return Provenance(
        source=fields["source"],
        source_md5=fields["source_md5"],
        md5=fields["md5"],
        converter=fields["converter"],
        source_format=fields.get("source_format", ""),
        quality=fields.get("quality", ""),
        source_removed=removed,
        source_url=fields.get("source_url", ""),
    )
