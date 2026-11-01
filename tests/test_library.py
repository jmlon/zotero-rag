"""Reading a whole library, user or group: what zotero-rag-web's indexer needs.

The CLI indexes one collection of a user library. A web deployment indexes a group
library from its root and keeps up with it incrementally, so the core must fetch
the whole library, what changed since a version, and what was removed, and must
cite items with links that are right for a group.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from zotero_rag_core.errors import ZoteroApiError
from zotero_rag_core.metadata import citation_url, web_url
from zotero_rag_core.planning import build_plan, library_membership
from zotero_rag_core.zotero_api import (
    ITEM_KEYS_PER_REQUEST,
    fetch_items_by_key,
    fetch_library_items,
    fetch_removed_item_keys,
    library_version,
)

GROUP_LINK = "https://www.zotero.org/groups/4711/items/P1"


def paper(key: str, collections: list[str] | None = None, link: str = "") -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": key,
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": f"Paper {key}",
            "date": "2020",
            "collections": collections or [],
        },
    }
    if link:
        entry["links"] = {"alternate": {"href": link, "type": "text/html"}}
    return entry


def pdf(key: str, parent: str) -> dict[str, Any]:
    return {
        "key": key,
        "data": {
            "key": key,
            "itemType": "attachment",
            "parentItem": parent,
            "contentType": "application/pdf",
            "linkMode": "imported_file",
            "filename": f"{key}.pdf",
            "md5": "x",
        },
    }


class FakeZotero:
    """Just enough of pyzotero: request methods return a query that `everything` runs."""

    def __init__(self, items: list[dict[str, Any]] | None = None, version: int = 42) -> None:
        self.items_by_key = {entry["key"]: entry for entry in items or []}
        self.version = version
        self.deleted_keys: list[str] = []
        self.trashed: list[dict[str, Any]] = []
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.fail: Exception | None = None

    def _ask(self, name: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if self.fail is not None:
            raise self.fail
        self.requests.append((name, kwargs))
        return name, kwargs

    def items(self, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return self._ask("items", kwargs)

    def trash(self, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return self._ask("trash", kwargs)

    def deleted(self, **kwargs: Any) -> dict[str, list[str]]:
        self._ask("deleted", kwargs)
        return {"items": list(self.deleted_keys), "collections": []}

    def last_modified_version(self) -> int:
        self._ask("version", {})
        return self.version

    def everything(self, query: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
        name, kwargs = query
        if name == "trash":
            return list(self.trashed)
        if "itemKey" in kwargs:
            keys = kwargs["itemKey"].split(",")
            return [self.items_by_key[key] for key in keys if key in self.items_by_key]
        return list(self.items_by_key.values())


# --- fetching -------------------------------------------------------------------


def test_the_whole_library_is_fetched_without_since() -> None:
    zot = FakeZotero([paper("P1"), pdf("A1", "P1")])
    assert {entry["key"] for entry in fetch_library_items(zot)} == {"P1", "A1"}  # type: ignore[arg-type]
    assert zot.requests == [("items", {})]


def test_an_incremental_fetch_passes_since() -> None:
    zot = FakeZotero([paper("P1")])
    fetch_library_items(zot, since=17)  # type: ignore[arg-type]
    assert zot.requests == [("items", {"since": 17})]


def test_the_library_version_is_read_from_the_api() -> None:
    assert library_version(FakeZotero(version=1234)) == 1234  # type: ignore[arg-type]


def test_items_are_fetched_by_key_in_batches_the_api_accepts() -> None:
    keys = [f"K{n:03d}" for n in range(ITEM_KEYS_PER_REQUEST + 1)]
    zot = FakeZotero([paper(key) for key in keys])
    found = fetch_items_by_key(zot, [*keys, keys[0]])  # type: ignore[arg-type]
    assert sorted(entry["key"] for entry in found) == keys
    sizes = [len(kwargs["itemKey"].split(",")) for _, kwargs in zot.requests]
    assert sizes == [ITEM_KEYS_PER_REQUEST, 1]


def test_no_keys_means_no_request() -> None:
    zot = FakeZotero()
    assert fetch_items_by_key(zot, []) == []  # type: ignore[arg-type]
    assert zot.requests == []


def test_removed_items_include_the_trash() -> None:
    """A trashed item is not in `/deleted`; it only stops appearing in `/items`."""
    zot = FakeZotero()
    zot.deleted_keys = ["GONE"]
    zot.trashed = [paper("BINNED")]
    assert fetch_removed_item_keys(zot, since=5) == {"GONE", "BINNED"}  # type: ignore[arg-type]
    assert ("deleted", {"since": 5}) in zot.requests
    assert ("trash", {"since": 5}) in zot.requests


@pytest.mark.parametrize(
    "call",
    [
        lambda zot: fetch_library_items(zot),
        lambda zot: fetch_items_by_key(zot, ["K"]),
        lambda zot: fetch_removed_item_keys(zot, since=1),
    ],
)
def test_a_network_failure_is_reported_as_a_zotero_error(call: Any) -> None:
    zot = FakeZotero()
    zot.fail = httpx.ConnectError("no route to host")
    with pytest.raises(ZoteroApiError, match="no route to host"):
        call(zot)


# --- citation links ----------------------------------------------------------------


def test_web_url_reads_the_api_link() -> None:
    assert web_url(paper("P1", link=GROUP_LINK)) == GROUP_LINK
    assert web_url(paper("P1")) == ""
    assert web_url({"links": "not a mapping"}) == ""


def test_a_user_library_is_cited_inside_the_first_collection() -> None:
    """The CLI's long-standing URL, unchanged when a username is known."""
    entry = paper("P1", ["C1"], link="https://www.zotero.org/someone/items/P1")
    assert citation_url("someone", entry) == (
        "https://www.zotero.org/someone/collections/C1/items/P1/collection"
    )


def test_without_a_username_the_api_link_is_cited() -> None:
    assert citation_url("", paper("P1", ["C1"], link=GROUP_LINK)) == GROUP_LINK


def test_a_group_plan_cites_the_api_links() -> None:
    plan = build_plan([("Group", [paper("P1", link=GROUP_LINK), pdf("A1", "P1")])])
    assert [entry.zotero_url for entry in plan.attachments] == [GROUP_LINK]


def test_a_user_plan_still_builds_urls_from_the_username() -> None:
    plan = build_plan([("Writing", [paper("P1", ["C1"]), pdf("A1", "P1")])], "someone")
    assert plan.attachments[0].zotero_url.startswith("https://www.zotero.org/someone/")


# --- planning a whole library -------------------------------------------------------


def test_library_membership_comes_from_each_items_own_collections() -> None:
    items = [paper("P1", ["C2", "C1"]), paper("P2"), pdf("A1", "P1")]
    assert library_membership(items) == {"P1": ["C1", "C2"], "P2": []}


def test_a_whole_library_plan_includes_unfiled_items() -> None:
    items = [paper("P1", ["C1"]), pdf("A1", "P1"), paper("P2"), pdf("A2", "P2")]
    plan = build_plan([("Library", items)], membership=library_membership(items))
    by_item = {entry.item_key: entry for entry in plan.attachments}
    assert set(by_item) == {"P1", "P2"}
    assert by_item["P1"].collection_keys == ["C1"]
    assert by_item["P2"].collection_keys == []


def test_the_item_type_is_planned_and_stored() -> None:
    plan = build_plan([("Library", [paper("P1"), pdf("A1", "P1")])])
    assert plan.attachments[0].item_type == "journalArticle"
    assert plan.attachments[0].chunk_metadata(1, 0)["item_type"] == "journalArticle"
