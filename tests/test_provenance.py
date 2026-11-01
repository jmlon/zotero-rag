"""The provenance note that tells a run which transcripts are its own (PLAN-markdown, Phase 2)."""

from __future__ import annotations

from datetime import date

import pytest

from zotero_rag_core.provenance import HEADER, Provenance, parse

MD5_A = "9e107d9d372bb6826bd81d3542a419d6"
MD5_B = "e4d909c290d0fb1ca068ffaddf22cbd0"


def record(**changes: object) -> Provenance:
    values: dict[str, object] = {
        "source": "ABCD2345",
        "source_md5": MD5_A,
        "md5": MD5_B,
        "converter": "anydoc 0.2.4",
    }
    values.update(changes)
    return Provenance.model_validate(values)


def test_round_trip() -> None:
    original = record(quality="pass recall=0.96 worst=0.93@12")

    assert parse(original.to_note()) == original


def test_round_trip_after_a_shrink() -> None:
    original = record(source_removed=date(2026, 11, 2), source_url="https://example.org/a.pdf")

    note = original.to_note()

    assert "source_removed=2026-11-02" in note
    assert parse(note) == original


def test_the_note_is_readable_by_a_person() -> None:
    assert record().to_note().splitlines() == [
        HEADER,
        "source=ABCD2345",
        f"source_md5={MD5_A}",
        f"md5={MD5_B}",
        "converter=anydoc 0.2.4",
    ]


def test_a_note_the_zotero_client_turned_into_html_still_parses() -> None:
    # Opening a note in the desktop client rewrites it as HTML paragraphs.
    html = "".join(f"<p>{line}</p>" for line in record().to_note().splitlines())

    assert parse(f'<div data-schema-version="9">{html}</div>') == record()


def test_unknown_keys_are_ignored_for_forward_compatibility() -> None:
    assert parse(record().to_note() + "\nfuture_field=whatever") == record()


@pytest.mark.parametrize(
    "note",
    [
        None,
        "",
        "Some note the user wrote about this file",
        record().to_note().replace(HEADER, "zotero-rag:markdown v2"),
        record().to_note().replace("source=ABCD2345", "source=abc"),
        record().to_note().replace(f"md5={MD5_B}", "md5=not-a-checksum"),
        record().to_note().replace("converter=anydoc 0.2.4", "converter="),
        record().to_note() + "\nsource_removed=yesterday",
        "\n".join(record().to_note().splitlines()[:3]),
    ],
)
def test_anything_else_is_not_ours(note: str | None) -> None:
    assert parse(note) is None


def test_the_first_value_of_a_repeated_key_wins() -> None:
    note = record().to_note() + "\nsource=ZZZZ9999"

    parsed = parse(note)

    assert parsed is not None
    assert parsed.source == "ABCD2345"
