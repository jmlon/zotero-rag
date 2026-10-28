"""The folder tree as scopes: what each node searches (PRD-CursesUI §4.2).

Selecting a folder searches it and every folder below it. The root, the pinned
collection, searches the whole store with no filter at all: it stays exactly the
`zotero-rag query` search, and it is the one scope that still works for a store
indexed before folder membership was recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zotero_rag.snapshot import CollectionSnapshot
from zotero_rag.zotero_api import CollectionNode

#: The root label when a whole-library index has several top-level collections, or
#: when there is no snapshot and no pinned collection name to show instead.
LIBRARY_LABEL = "Library"


@dataclass(frozen=True)
class ScopeChoice:
    """One selectable folder: how it is labelled, and the keys it searches."""

    #: Folder names from the root down, e.g. `Networking/Transport`.
    path: str
    #: `None` means no filter: the whole store.
    keys: frozenset[str] | None


@dataclass
class ScopeNode:
    choice: ScopeChoice
    children: list[ScopeNode] = field(default_factory=list)


def _folder(node: CollectionNode, parent_path: str) -> ScopeNode:
    path = f"{parent_path}/{node.name}" if parent_path else node.name
    keys = frozenset(descendant.key for descendant, _depth in node.walk())
    return ScopeNode(ScopeChoice(path, keys), [_folder(child, path) for child in node.children])


def build_scope_tree(snapshot: CollectionSnapshot | None, root_name: str | None) -> ScopeNode:
    """The tree to draw, rooted at an unfiltered node.

    Without a snapshot, folder filtering is unavailable and the tree is the root
    alone (§4.3.2). A pinned collection is the root itself. A whole-library index
    has several top-level collections, so they sit under a `Library` root.
    """
    if snapshot is None:
        return ScopeNode(ScopeChoice(root_name or LIBRARY_LABEL, None))
    if len(snapshot.roots) == 1:
        root = snapshot.roots[0]
        return ScopeNode(
            ScopeChoice(root.name, None), [_folder(child, root.name) for child in root.children]
        )
    return ScopeNode(
        ScopeChoice(LIBRARY_LABEL, None), [_folder(root, "") for root in snapshot.roots]
    )
