"""Filtering by folder, year and tag (`store.Filters`).

Two implementations of one rule must agree: `Filters.matches`, which keyword
search applies in Python, and the `where` clause the Chroma store sends. Chroma
fails silently on a wrong filter (see test_scope.py), so the agreement is checked
against a real store over every combination of conditions, not by reading clauses.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest

from zotero_rag_core.metadata import item_tags, year_number
from zotero_rag_core.planning import PlannedAttachment
from zotero_rag_core.retrieval import BM25Searcher
from zotero_rag_core.store import YEAR_UNKNOWN, Filters, passes
from zotero_rag_core.stores.chroma import ChromaStore, filters_where

#: attachment → metadata beyond the basics. `None` leaves the field out, as in a
#: chunk written before it existed.
LIBRARY: dict[str, dict[str, Any]] = {
    "OLD": {"year_int": 2010, "tags": ["RAG"], "collection_keys": ["K1"]},
    "MID": {"year_int": 2018, "tags": ["RAG", "LLM"], "collection_keys": ["K2"]},
    "NEW": {"year_int": 2023, "tags": None, "collection_keys": ["K1", "K2"]},
    "UNDATED": {"year_int": YEAR_UNKNOWN, "tags": ["LLM"], "collection_keys": ["K2"]},
    "LEGACY": {"year_int": None, "tags": None, "collection_keys": None},
}


def chunk_metadata(key: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"attachment_key": key, "title": key, "page": 1, "chunk_index": 0}
    meta.update({field: value for field, value in LIBRARY[key].items() if value is not None})
    return meta


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> ChromaStore:
    store = ChromaStore(tmp_path_factory.mktemp("filters") / "chroma")
    store.add(
        ids=[f"{key}:1:0" for key in LIBRARY],
        embeddings=[[1.0, float(position), 0.0] for position in range(len(LIBRARY))],
        documents=[f"Retrieval paper {key}." for key in LIBRARY],
        metadatas=[chunk_metadata(key) for key in LIBRARY],
    )
    return store


def found(chunks: list[Any]) -> set[str]:
    return {chunk.metadata["attachment_key"] for chunk in chunks}


def expected(filters: Filters | None) -> set[str]:
    return {key for key in LIBRARY if passes(chunk_metadata(key), filters)}


# --- the rule itself ----------------------------------------------------------


def test_a_year_range_is_inclusive() -> None:
    assert expected(Filters(year_min=2010, year_max=2018, include_undated=False)) == {"OLD", "MID"}


def test_an_open_ended_range_excludes_undated_items_unless_asked() -> None:
    """Undated is stored as 0, which a bare `year <= 2018` would let through."""
    assert expected(Filters(year_max=2018, include_undated=False)) == {"OLD", "MID"}
    assert expected(Filters(year_max=2018)) == {"OLD", "MID", "UNDATED"}


def test_a_chunk_without_the_field_fails_a_condition_on_it() -> None:
    assert "LEGACY" not in expected(Filters(year_min=2000))
    assert "LEGACY" not in expected(Filters(tags=frozenset({"RAG"})))
    assert "LEGACY" in expected(None)


def test_tags_match_any_of_the_chosen() -> None:
    assert expected(Filters(tags=frozenset({"RAG"}))) == {"OLD", "MID"}
    assert expected(Filters(tags=frozenset({"RAG", "LLM"}))) == {"OLD", "MID", "UNDATED"}


def test_conditions_combine_with_and() -> None:
    filters = Filters(collection_keys=frozenset({"K2"}), tags=frozenset({"LLM"}), year_min=2015)
    assert expected(filters) == {"MID", "UNDATED"}


def test_an_empty_set_selects_nothing() -> None:
    assert Filters(tags=frozenset()).selects_nothing
    assert Filters(collection_keys=frozenset()).selects_nothing
    assert not Filters().selects_nothing


def test_reversed_years_are_refused() -> None:
    with pytest.raises(ValueError, match="after year_max"):
        Filters(year_min=2020, year_max=2010)


def test_no_conditions_means_no_where_clause() -> None:
    """So an unfiltered query stays exactly the call it always was."""
    assert filters_where(Filters()) is None


# --- Chroma agrees with the rule ------------------------------------------------

COLLECTIONS: list[frozenset[str] | None] = [
    None,
    frozenset({"K1"}),
    frozenset({"K1", "K2"}),
    frozenset(),
]
TAGS: list[frozenset[str] | None] = [None, frozenset({"RAG"}), frozenset({"RAG", "LLM"})]
YEARS: list[tuple[int | None, int | None]] = [
    (None, None),
    (2015, None),
    (None, 2018),
    (2010, 2018),
]


def every_filter() -> list[Filters]:
    combinations = itertools.product(COLLECTIONS, TAGS, YEARS, [True, False])
    return [
        Filters(collection_keys=keys, tags=tags, year_min=low, year_max=high, include_undated=u)
        for keys, tags, (low, high), u in combinations
    ]


@pytest.mark.parametrize("filters", every_filter(), ids=repr)
def test_chroma_returns_exactly_what_the_rule_accepts(store: ChromaStore, filters: Filters) -> None:
    assert found(store.query([1.0, 0.0, 0.0], len(LIBRARY), filters)) == expected(filters)


@pytest.mark.parametrize("filters", every_filter(), ids=repr)
def test_keyword_search_returns_exactly_what_the_rule_accepts(
    store: ChromaStore, filters: Filters
) -> None:
    index = BM25Searcher.from_store(store)
    assert found(index.search("retrieval", len(LIBRARY), filters)) == expected(filters)


def test_the_filter_applies_before_the_cut(store: ChromaStore) -> None:
    """UNDATED is the farthest chunk; filtered to it, a top-1 search must still find it."""
    filters = Filters(tags=frozenset({"LLM"}), year_max=2000)
    assert found(store.query([1.0, 0.0, 0.0], 1, filters)) == {"UNDATED"}


# --- what indexing writes -------------------------------------------------------


def test_item_tags_are_sorted_and_deduplicated() -> None:
    data = {"tags": [{"tag": "RAG", "type": 1}, {"tag": " LLM "}, {"tag": "RAG"}, {"tag": ""}]}
    assert item_tags(data) == ["LLM", "RAG"]
    assert item_tags({}) == []


def test_year_number() -> None:
    assert year_number("2019") == 2019
    assert year_number("") == YEAR_UNKNOWN


def planned(**fields: Any) -> PlannedAttachment:
    base: dict[str, Any] = {
        "attachment_key": "A",
        "item_key": "I",
        "title": "t",
        "authors": "",
        "year": "",
        "collection_name": "c",
    }
    return PlannedAttachment(**(base | fields))


def test_chunk_metadata_carries_year_and_tags() -> None:
    metadata = planned(year="2019", tags=["LLM", "RAG"]).chunk_metadata(1, 0)
    assert metadata["year_int"] == 2019
    assert metadata["tags"] == ["LLM", "RAG"]


def test_chunk_metadata_of_an_undated_untagged_item() -> None:
    """Chroma rejects an empty list, so `tags` is left out rather than written empty."""
    metadata = planned().chunk_metadata(1, 0)
    assert metadata["year_int"] == YEAR_UNKNOWN
    assert "tags" not in metadata


def test_planning_takes_tags_from_the_parent_item(tmp_path: Path) -> None:
    from zotero_rag_core.planning import build_plan

    parent = {
        "key": "P1",
        "data": {
            "key": "P1",
            "itemType": "journalArticle",
            "title": "Paper",
            "date": "2021",
            "tags": [{"tag": "RAG"}],
        },
    }
    attachment = {
        "key": "A1",
        "data": {
            "key": "A1",
            "itemType": "attachment",
            "parentItem": "P1",
            "contentType": "application/pdf",
            "linkMode": "imported_file",
            "filename": "paper.pdf",
            "md5": "x",
        },
    }
    plan = build_plan([("Writing", [parent, attachment])])
    assert [entry.tags for entry in plan.attachments] == [["RAG"]]
    assert plan.attachments[0].chunk_metadata(1, 0)["year_int"] == 2021
