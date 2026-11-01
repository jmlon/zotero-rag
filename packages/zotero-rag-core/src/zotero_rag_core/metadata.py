"""Bibliographic metadata extracted from raw Zotero item payloads.

Ported from the reference `zotero_ingest.py`, which had to cope with a decade of
messy real-world entries: creators with only a `name`, non-author creator types,
and dates in several formats.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 2012-07, 2021-10-05, 2023-11-22T15:34:18Z
_YEAR_FIRST = re.compile(r"(?P<year>\d{4})(-\d{1,2})?(-\d{1,2})?")
# 4/2022
_MONTH_SLASH_YEAR = re.compile(r"\d{1,2}/(?P<year>\d{4})")
# 22-11-2023
_DAY_MONTH_YEAR = re.compile(r"\d{1,2}-\d{1,2}-(?P<year>\d{4})")

# Creator types that count as an author for citation purposes.
_CITED_CREATOR_TYPES = frozenset({"author", "editor"})


def creator_name(creator: dict[str, Any]) -> str:
    """Format one creator as `Last, First`, or its single `name` field."""
    last, first = creator.get("lastName"), creator.get("firstName")
    if last and first:
        return f"{last}, {first}"
    if last:
        return str(last)
    name = creator.get("name")
    return str(name) if name else ""


def authors_str(item: dict[str, Any]) -> str:
    """Semicolon-separated authors and editors of an item's `creators` list.

    Other creator types (sponsor, contributor, ...) are ignored rather than
    dropped silently at the item level: the item still gets its remaining
    authors, and an item with none yields an empty string.
    """
    creators = item.get("creators")
    if not creators:
        return ""

    names: list[str] = []
    for creator in creators:
        if not isinstance(creator, dict):
            continue
        creator_type = creator.get("creatorType")
        if creator_type not in _CITED_CREATOR_TYPES:
            logger.debug("ignoring creator type %r", creator_type)
            continue
        name = creator_name(creator)
        if name:
            names.append(name)
        else:
            logger.debug("creator with no usable name: %r", creator)
    return "; ".join(names)


def parse_year(date_value: str | None) -> str:
    """Extract a four-digit year from Zotero's free-form `date` field.

    Returns an empty string when no year can be found — an unparsable date must
    never abort indexing.
    """
    if not date_value:
        return ""
    text = str(date_value).strip()
    for pattern in (_YEAR_FIRST, _MONTH_SLASH_YEAR, _DAY_MONTH_YEAR):
        match = pattern.match(text)
        if match:
            return match.group("year")
    return ""


def zotero_url(username: str, item_key: str, collections: list[str] | None) -> str:
    """Web URL for an item, inside its first collection when it has one."""
    if not username:
        return ""
    if collections:
        return (
            f"https://www.zotero.org/{username}/collections/{collections[0]}"
            f"/items/{item_key}/collection"
        )
    return f"https://www.zotero.org/{username}/items/{item_key}"


def year_number(year: str) -> int:
    """`parse_year`'s result as a number; `store.YEAR_UNKNOWN` (0) when there is none."""
    return int(year) if year.isdigit() else 0


def item_tags(data: dict[str, Any]) -> list[str]:
    """An item's tags, sorted and de-duplicated.

    Zotero sends `tags` as `[{"tag": "RAG", "type": 1}, ...]`: type 0 is a tag the
    user added, 1 one imported automatically. Both count, as in Zotero's own tag
    selector.
    """
    names = {
        str(entry.get("tag")).strip()
        for entry in data.get("tags") or []
        if isinstance(entry, dict) and str(entry.get("tag") or "").strip()
    }
    return sorted(names)


def web_url(entry: dict[str, Any]) -> str:
    """The item's page in the zotero.org web library, as the API links it.

    Every item payload carries `links.alternate`, which points into the right
    library whoever owns it: `/<username>/items/<key>` for a user library and
    `/groups/<id or name>/items/<key>` for a group. Empty when the payload has no
    links, as hand-built test payloads usually do not.
    """
    links = entry.get("links")
    alternate = links.get("alternate") if isinstance(links, dict) else None
    href = alternate.get("href") if isinstance(alternate, dict) else None
    return str(href) if href else ""


def citation_url(username: str, entry: dict[str, Any]) -> str:
    """The URL a citation links to.

    With a `username`, the item inside its first collection (`zotero_url`), as the
    CLI has always cited a user library. Without one, the API's own link
    (`web_url`), which is the only correct choice for a group library: the
    username behind an API key is not the library's owner.
    """
    data = entry.get("data") or {}
    key = str(data.get("key") or entry.get("key") or "")
    if username and key:
        return zotero_url(username, key, data.get("collections"))
    return web_url(entry)
