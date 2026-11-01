"""Collection-hierarchy construction from raw Zotero Web API payloads.

The payloads below are shaped like real `pyzotero.collections()` responses (the
fields this code reads: `data.key`, `data.name`, `data.parentCollection`, and
`meta.numItems`). No pyzotero client is mocked — `build_collection_tree` is a
pure function over the decoded JSON.
"""

from __future__ import annotations

from typing import Any

from zotero_rag_core.zotero_api import CollectionNode, build_collection_tree, count_collections


def collection(
    key: str, name: str, parent: str | bool = False, num_items: int = 0
) -> dict[str, Any]:
    return {
        "key": key,
        "version": 1,
        "meta": {"numCollections": 0, "numItems": num_items},
        "data": {"key": key, "version": 1, "name": name, "parentCollection": parent},
    }


def names(nodes: list[CollectionNode]) -> list[str]:
    return [node.name for node in nodes]


def test_empty_library_yields_no_roots() -> None:
    assert build_collection_tree([]) == []
    assert count_collections([]) == 0


def test_flat_collections_become_roots_sorted_by_name() -> None:
    raw = [
        collection("BBB", "Writing", num_items=4),
        collection("AAA", "anonymization", num_items=2),
        collection("CCC", "Legal"),
    ]
    roots = build_collection_tree(raw)
    # Sorting is case-insensitive, so "anonymization" precedes "Legal".
    assert names(roots) == ["anonymization", "Legal", "Writing"]
    assert roots[2].key == "BBB"
    assert roots[2].num_items == 4
    assert count_collections(roots) == 3


def test_nested_collections_are_attached_to_their_parent() -> None:
    raw = [
        collection("ROOT", "Research", num_items=1),
        collection("KID1", "Data Trust", parent="ROOT", num_items=5),
        collection("KID2", "Anonymization", parent="ROOT", num_items=3),
        collection("GRAND", "Legal", parent="KID1", num_items=2),
    ]
    roots = build_collection_tree(raw)
    assert names(roots) == ["Research"]
    research = roots[0]
    assert names(research.children) == ["Anonymization", "Data Trust"]
    data_trust = research.children[1]
    assert names(data_trust.children) == ["Legal"]
    assert count_collections(roots) == 4


def test_total_items_sums_descendants() -> None:
    raw = [
        collection("ROOT", "Research", num_items=1),
        collection("KID1", "Data Trust", parent="ROOT", num_items=5),
        collection("GRAND", "Legal", parent="KID1", num_items=2),
    ]
    root = build_collection_tree(raw)[0]
    assert root.num_items == 1
    assert root.total_items == 8


def test_walk_is_depth_first_with_depths() -> None:
    raw = [
        collection("ROOT", "Research"),
        collection("KID1", "Alpha", parent="ROOT"),
        collection("GRAND", "Deep", parent="KID1"),
        collection("KID2", "Beta", parent="ROOT"),
    ]
    root = build_collection_tree(raw)[0]
    assert [(node.name, depth) for node, depth in root.walk()] == [
        ("Research", 0),
        ("Alpha", 1),
        ("Deep", 2),
        ("Beta", 1),
    ]


def test_orphan_with_unknown_parent_is_surfaced_as_a_root() -> None:
    """A partial fetch must not silently swallow collections."""
    raw = [
        collection("ROOT", "Research"),
        collection("ORPHAN", "Stray", parent="MISSING"),
    ]
    roots = build_collection_tree(raw)
    assert names(roots) == ["Research", "Stray"]


def test_input_order_does_not_matter() -> None:
    """Children may arrive before their parents in a paginated fetch."""
    raw = [
        collection("GRAND", "Legal", parent="KID1"),
        collection("KID1", "Data Trust", parent="ROOT"),
        collection("ROOT", "Research"),
    ]
    roots = build_collection_tree(raw)
    assert names(roots) == ["Research"]
    assert names(roots[0].children[0].children) == ["Legal"]


def test_missing_metadata_is_tolerated() -> None:
    raw: list[dict[str, Any]] = [
        {"key": "NOMETA", "data": {"key": "NOMETA", "name": "Bare", "parentCollection": False}},
        {"key": "NONAME", "data": {"key": "NONAME", "parentCollection": False}, "meta": {}},
    ]
    roots = build_collection_tree(raw)
    assert names(roots) == ["(unnamed)", "Bare"]
    assert all(node.num_items == 0 for node in roots)


def test_entries_without_a_key_are_skipped() -> None:
    raw: list[dict[str, Any]] = [
        collection("GOOD", "Keeper"),
        {"data": {"name": "Broken", "parentCollection": False}},
    ]
    roots = build_collection_tree(raw)
    assert names(roots) == ["Keeper"]


def test_null_parent_is_treated_as_top_level() -> None:
    """Zotero uses False, but tolerate null/empty-string parents too."""
    raw = [
        collection("A", "WithNull", parent=""),
        {
            "key": "B",
            "meta": {"numItems": 0},
            "data": {"key": "B", "name": "WithNone", "parentCollection": None},
        },
    ]
    roots = build_collection_tree(raw)
    assert names(roots) == ["WithNone", "WithNull"]
