"""Bibliographic metadata extraction.

The payloads mirror real entries from the reference library, including the awkward
ones: institutional authors with only a `name`, non-author creator types, and the
several date formats Zotero accepts.
"""

from __future__ import annotations

from typing import Any

import pytest

from zotero_rag_core.metadata import authors_str, parse_year, zotero_url


def test_first_last_names_are_formatted_last_first() -> None:
    item = {
        "creators": [
            {"creatorType": "author", "firstName": "Clete A.", "lastName": "Kushida"},
            {"creatorType": "author", "firstName": "Deborah A.", "lastName": "Nichols"},
        ]
    }
    assert authors_str(item) == "Kushida, Clete A.; Nichols, Deborah A."


def test_institutional_author_with_only_a_name() -> None:
    item = {"creators": [{"creatorType": "author", "name": "Information Commissioner's Office"}]}
    assert authors_str(item) == "Information Commissioner's Office"


def test_editors_are_included() -> None:
    item = {
        "creators": [
            {"creatorType": "editor", "lastName": "Knuth", "firstName": "Donald"},
            {"creatorType": "author", "lastName": "Lamport", "firstName": "Leslie"},
        ]
    }
    assert authors_str(item) == "Knuth, Donald; Lamport, Leslie"


def test_other_creator_types_are_ignored_without_losing_authors() -> None:
    item = {
        "creators": [
            {"creatorType": "sponsor", "name": "The European Parliament"},
            {"creatorType": "author", "lastName": "Doe", "firstName": "Jane"},
        ]
    }
    assert authors_str(item) == "Doe, Jane"


def test_missing_or_empty_creators_yield_an_empty_string() -> None:
    assert authors_str({}) == ""
    assert authors_str({"creators": []}) == ""
    assert authors_str({"creators": [{"creatorType": "sponsor", "name": "Only a sponsor"}]}) == ""


def test_creator_without_a_usable_name_is_dropped() -> None:
    item: dict[str, Any] = {
        "creators": [
            {"creatorType": "author"},
            {"creatorType": "author", "lastName": "Solo"},
        ]
    }
    assert authors_str(item) == "Solo"


def test_malformed_creator_entries_do_not_crash() -> None:
    item = {"creators": ["not a dict", {"creatorType": "author", "name": "Real"}]}
    assert authors_str(item) == "Real"


@pytest.mark.parametrize(
    ("date_value", "expected"),
    [
        ("2012-07", "2012"),
        ("2021-10", "2021"),
        ("2021-10-05", "2021"),
        ("2023-11-22T15:34:18Z", "2023"),
        ("2019", "2019"),
        ("4/2022", "2022"),
        ("12/1999", "1999"),
        ("22-11-2023", "2023"),
        ("", ""),
        ("   ", ""),
        (None, ""),
        ("n.d.", ""),
        ("forthcoming", ""),
    ],
)
def test_parse_year_handles_zotero_date_formats(date_value: str | None, expected: str) -> None:
    assert parse_year(date_value) == expected


def test_zotero_url_points_into_the_first_collection() -> None:
    url = zotero_url("jmlondono", "ABSY8J4R", ["4V3RZW4U"])
    assert url == (
        "https://www.zotero.org/jmlondono/collections/4V3RZW4U/items/ABSY8J4R/collection"
    )


def test_zotero_url_without_a_collection() -> None:
    assert zotero_url("jmlondono", "ABSY8J4R", []) == (
        "https://www.zotero.org/jmlondono/items/ABSY8J4R"
    )


def test_zotero_url_without_a_username_is_empty() -> None:
    assert zotero_url("", "ABSY8J4R", ["4V3RZW4U"]) == ""
