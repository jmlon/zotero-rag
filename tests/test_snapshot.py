"""The saved collection tree that `zotero-rag-ui` draws (PRD-CursesUI §4.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from zotero_rag.snapshot import (
    COLLECTIONS_FILENAME,
    SNAPSHOT_VERSION,
    read_snapshot,
    write_snapshot,
)
from zotero_rag_core.zotero_api import CollectionNode, scope_tree


def tree() -> CollectionNode:
    return CollectionNode(
        key="ROOT",
        name="Networking",
        children=[
            CollectionNode(key="R1", name="Routing"),
            CollectionNode(
                key="T1", name="Transport", children=[CollectionNode(key="Q", name="QUIC")]
            ),
        ],
    )


def test_round_trip_keeps_the_whole_tree(tmp_path: Path) -> None:
    write_snapshot(tmp_path, [tree()])
    snapshot = read_snapshot(tmp_path)
    assert snapshot is not None
    assert snapshot.roots == [tree()]
    assert snapshot.all_keys() == {"ROOT", "R1", "T1", "Q"}
    assert snapshot.version == SNAPSHOT_VERSION


def test_rewriting_replaces_the_snapshot_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    write_snapshot(tmp_path, [tree()])
    write_snapshot(tmp_path, [CollectionNode(key="ROOT", name="Networking")])
    snapshot = read_snapshot(tmp_path)
    assert snapshot is not None and snapshot.all_keys() == {"ROOT"}
    assert [path.name for path in tmp_path.iterdir()] == [COLLECTIONS_FILENAME]


def test_a_missing_snapshot_reads_as_none(tmp_path: Path) -> None:
    assert read_snapshot(tmp_path) is None


@pytest.mark.parametrize("content", ["{not json", '{"roots": "nope"}', '{"version": 99}'])
def test_an_unusable_snapshot_reads_as_none(tmp_path: Path, content: str) -> None:
    """The remedy is the same as for a missing one: run `index` again."""
    (tmp_path / COLLECTIONS_FILENAME).write_text(content, encoding="utf-8")
    assert read_snapshot(tmp_path) is None


def test_a_non_recursive_index_saves_only_its_root() -> None:
    """Subfolders were not indexed, so the tree must not offer them."""
    assert scope_tree(tree(), recursive=False).children == []
    assert scope_tree(tree(), recursive=True) == tree()
