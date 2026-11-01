"""Collection path/key resolution.

The fixture mirrors the shape of the real library, including the duplicate names
that appear at different levels there ("Blockchain", "Crypto", "Routing").
"""

from __future__ import annotations

import pytest

from zotero_rag_core.errors import AmbiguousCollectionError, CollectionNotFoundError
from zotero_rag_core.zotero_api import (
    CollectionNode,
    build_collection_tree,
    collection_scope,
    find_collection_by_key,
    library_scope,
    resolve_collection_path,
    split_collection_path,
)


@pytest.fixture
def roots() -> list[CollectionNode]:
    raw = [
        {"key": "BC", "meta": {"numItems": 22}, "data": {"key": "BC", "name": "Blockchain"}},
        {
            "key": "BCX",
            "meta": {"numItems": 1},
            "data": {"key": "BCX", "name": "Crypto", "parentCollection": "BC"},
        },
        {"key": "DM", "meta": {"numItems": 10}, "data": {"key": "DM", "name": "DataManagement"}},
        {
            "key": "DMB",
            "meta": {"numItems": 18},
            "data": {"key": "DMB", "name": "Blockchain", "parentCollection": "DM"},
        },
        {
            "key": "DMC",
            "meta": {"numItems": 11},
            "data": {"key": "DMC", "name": "Crypto", "parentCollection": "DM"},
        },
        {
            "key": "DMT",
            "meta": {"numItems": 175},
            "data": {"key": "DMT", "name": "Data Trust", "parentCollection": "DM"},
        },
        {
            "key": "DMTP",
            "meta": {"numItems": 4},
            "data": {"key": "DMTP", "name": "PET", "parentCollection": "DMT"},
        },
        {"key": "WR", "meta": {"numItems": 6}, "data": {"key": "WR", "name": "Writting"}},
    ]
    return build_collection_tree(raw)


def test_split_tolerates_surrounding_and_repeated_slashes() -> None:
    assert split_collection_path("A/B") == ["A", "B"]
    assert split_collection_path("/A/B/") == ["A", "B"]
    assert split_collection_path("//A//B//") == ["A", "B"]
    assert split_collection_path(" A / B ") == ["A", "B"]
    assert split_collection_path("/") == []


def test_resolve_top_level_collection(roots: list[CollectionNode]) -> None:
    assert resolve_collection_path(roots, "Writting").key == "WR"


def test_resolve_nested_path(roots: list[CollectionNode]) -> None:
    assert resolve_collection_path(roots, "DataManagement/Data Trust").key == "DMT"
    assert resolve_collection_path(roots, "DataManagement/Data Trust/PET").key == "DMTP"


def test_resolution_is_case_insensitive(roots: list[CollectionNode]) -> None:
    assert resolve_collection_path(roots, "datamanagement/data trust").key == "DMT"


def test_leading_slash_is_accepted(roots: list[CollectionNode]) -> None:
    """The PRD writes paths as /path/to/collection."""
    assert resolve_collection_path(roots, "/DataManagement/Data Trust").key == "DMT"


def test_same_name_at_different_levels_is_not_ambiguous(roots: list[CollectionNode]) -> None:
    """'Blockchain' exists at the top level and under DataManagement."""
    assert resolve_collection_path(roots, "Blockchain").key == "BC"
    assert resolve_collection_path(roots, "DataManagement/Blockchain").key == "DMB"


def test_unknown_segment_lists_the_alternatives(roots: list[CollectionNode]) -> None:
    with pytest.raises(CollectionNotFoundError) as excinfo:
        resolve_collection_path(roots, "Nonexistent")
    message = str(excinfo.value)
    assert "Nonexistent" in message
    assert "Writting" in message  # the available options are shown


def test_unknown_nested_segment_names_its_parent(roots: list[CollectionNode]) -> None:
    with pytest.raises(CollectionNotFoundError, match="DataManagement"):
        resolve_collection_path(roots, "DataManagement/Missing")


def test_empty_path_is_rejected(roots: list[CollectionNode]) -> None:
    with pytest.raises(CollectionNotFoundError):
        resolve_collection_path(roots, "/")


def test_ambiguous_sibling_names_report_both_keys() -> None:
    raw = [
        {"key": "A1", "data": {"key": "A1", "name": "Legal"}},
        {"key": "A2", "data": {"key": "A2", "name": "Legal"}},
    ]
    with pytest.raises(AmbiguousCollectionError) as excinfo:
        resolve_collection_path(build_collection_tree(raw), "Legal")
    message = str(excinfo.value)
    assert "A1" in message and "A2" in message
    assert "--collection-key" in message


def test_find_by_key_searches_the_whole_hierarchy(roots: list[CollectionNode]) -> None:
    assert find_collection_by_key(roots, "DMTP").name == "PET"
    assert find_collection_by_key(roots, "WR").name == "Writting"


def test_find_by_unknown_key_raises(roots: list[CollectionNode]) -> None:
    with pytest.raises(CollectionNotFoundError, match="NOPE"):
        find_collection_by_key(roots, "NOPE")


def test_recursive_scope_includes_descendants(roots: list[CollectionNode]) -> None:
    node = find_collection_by_key(roots, "DM")
    keys = {collection.key for collection in collection_scope(node, recursive=True)}
    assert keys == {"DM", "DMB", "DMC", "DMT", "DMTP"}


def test_non_recursive_scope_is_just_the_collection(roots: list[CollectionNode]) -> None:
    node = find_collection_by_key(roots, "DM")
    assert [c.key for c in collection_scope(node, recursive=False)] == ["DM"]


def test_leaf_scope_is_itself(roots: list[CollectionNode]) -> None:
    node = find_collection_by_key(roots, "WR")
    assert [c.key for c in collection_scope(node, recursive=True)] == ["WR"]


def test_library_scope_covers_every_collection(roots: list[CollectionNode]) -> None:
    keys = {collection.key for collection in library_scope(roots)}
    assert keys == {"BC", "BCX", "DM", "DMB", "DMC", "DMT", "DMTP", "WR"}
