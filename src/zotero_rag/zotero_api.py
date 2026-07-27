"""Zotero Web API access and collection-hierarchy handling."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from pyzotero import zotero
from pyzotero.zotero_errors import PyZoteroError

from zotero_rag.config import Config
from zotero_rag.errors import ZoteroApiError

logger = logging.getLogger(__name__)


class CollectionNode(BaseModel):
    """A node in the Zotero collection hierarchy."""

    key: str
    name: str
    num_items: int = 0
    children: list[CollectionNode] = Field(default_factory=list)

    @property
    def total_items(self) -> int:
        """Item count for this collection plus all of its descendants."""
        return self.num_items + sum(child.total_items for child in self.children)

    def walk(self, depth: int = 0) -> list[tuple[CollectionNode, int]]:
        """Depth-first list of (node, depth) pairs, this node first."""
        pairs = [(self, depth)]
        for child in self.children:
            pairs.extend(child.walk(depth + 1))
        return pairs


def build_client(config: Config) -> zotero.Zotero:
    """Build an authenticated pyzotero client. Requires Zotero credentials."""
    config.global_.require_zotero()
    return zotero.Zotero(
        config.global_.zotero_library_id,
        config.global_.zotero_library_type,
        config.global_.zotero_api_key,
    )


def fetch_collections(zot: zotero.Zotero) -> list[dict[str, Any]]:
    """Fetch every collection in the library, following pagination."""
    try:
        return list(zot.everything(zot.collections()))
    except PyZoteroError as exc:
        raise ZoteroApiError(f"Could not fetch collections from Zotero: {exc}") from exc


def fetch_username(zot: zotero.Zotero) -> str:
    """The library owner's username, used to build zotero.org item URLs."""
    try:
        info = zot.key_info()
    except PyZoteroError as exc:
        raise ZoteroApiError(f"Could not verify the Zotero API key: {exc}") from exc
    username = info.get("username", "")
    return str(username)


def build_collection_tree(raw_collections: list[dict[str, Any]]) -> list[CollectionNode]:
    """Turn pyzotero's flat collection list into a sorted forest of roots.

    Each raw collection carries `data.parentCollection`, which is `False` for a
    top-level collection and a parent key otherwise. Collections whose parent is
    absent from the input (e.g. a partial fetch) are surfaced as roots rather
    than silently dropped.
    """
    nodes: dict[str, CollectionNode] = {}
    parents: dict[str, str | None] = {}

    for entry in raw_collections:
        data = entry.get("data", {})
        key = data.get("key") or entry.get("key")
        if not key:
            logger.warning("Skipping collection with no key: %r", entry)
            continue
        meta = entry.get("meta", {})
        nodes[key] = CollectionNode(
            key=key,
            name=data.get("name", "(unnamed)"),
            num_items=int(meta.get("numItems", 0) or 0),
        )
        parent = data.get("parentCollection")
        parents[key] = parent if isinstance(parent, str) and parent else None

    roots: list[CollectionNode] = []
    for key, node in nodes.items():
        parent_key = parents[key]
        if parent_key is None:
            roots.append(node)
        elif parent_key in nodes:
            nodes[parent_key].children.append(node)
        else:
            logger.warning(
                "Collection %r (%s) references unknown parent %s; treating as top level",
                node.name,
                key,
                parent_key,
            )
            roots.append(node)

    _sort_recursive(roots)
    return roots


def _sort_recursive(nodes: list[CollectionNode]) -> None:
    """Sort siblings by name, case-insensitively, at every level."""
    nodes.sort(key=lambda node: node.name.casefold())
    for node in nodes:
        _sort_recursive(node.children)


def count_collections(nodes: list[CollectionNode]) -> int:
    """Total number of collections in a forest."""
    return sum(len(node.walk()) for node in nodes)
