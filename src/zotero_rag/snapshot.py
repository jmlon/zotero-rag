"""The collection tree an index was built from, saved beside the store.

`index` writes it once the pass completes, and `zotero-rag-ui` reads it to draw the
folder tree with no Zotero API call. Its presence also tells the UI that every
planned chunk carries `collection_keys`, which subtree filtering depends on, because
it is only written after `refresh_collection_keys` has run (PRD-CursesUI §4.1).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from zotero_rag.errors import ZoteroRagError
from zotero_rag.zotero_api import CollectionNode, find_collection_by_key

logger = logging.getLogger(__name__)

COLLECTIONS_FILENAME = "collections.json"
SNAPSHOT_VERSION = 1


class SnapshotError(ZoteroRagError):
    """The collection snapshot could not be written."""


class CollectionSnapshot(BaseModel):
    """The indexed collections, as a forest: one root for a pinned collection."""

    version: int = SNAPSHOT_VERSION
    written_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    roots: list[CollectionNode] = Field(default_factory=list)

    def all_keys(self) -> set[str]:
        return {node.key for root in self.roots for node, _depth in root.walk()}

    def subtree_keys(self, key: str) -> frozenset[str]:
        """The scope for a selected folder: its own key and every descendant's.

        Expanded here, at query time, rather than stored on each chunk, so that
        rearranging folders only ever changes membership, never the store's layout.
        Raises `CollectionNotFoundError` for a key the snapshot does not hold.
        """
        node = find_collection_by_key(self.roots, key)
        return frozenset(descendant.key for descendant, _depth in node.walk())


def write_snapshot(index_dir: Path, roots: list[CollectionNode]) -> CollectionSnapshot:
    """Save the indexed tree, replacing any earlier snapshot in one step.

    Writing to a temporary file and renaming it means a reader never sees half a
    file, and an interrupted write leaves the previous snapshot in place.
    """
    snapshot = CollectionSnapshot(roots=roots)
    target = index_dir / COLLECTIONS_FILENAME
    partial = target.with_name(f".{COLLECTIONS_FILENAME}.tmp")
    try:
        partial.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        partial.replace(target)
    except OSError as exc:
        raise SnapshotError(f"Cannot write {target}: {exc}") from exc
    return snapshot


def read_snapshot(index_dir: Path) -> CollectionSnapshot | None:
    """The saved tree, or `None` when there is none or it cannot be used.

    An unreadable or newer-format file is treated like a missing one, because the
    remedy is the same either way: run `index` again.
    """
    path = index_dir / COLLECTIONS_FILENAME
    try:
        snapshot = CollectionSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as exc:
        logger.warning("ignoring unreadable %s: %s", path, exc)
        return None
    if snapshot.version != SNAPSHOT_VERSION:
        logger.warning("ignoring %s: format %d is not supported", path, snapshot.version)
        return None
    return snapshot
