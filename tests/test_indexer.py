"""The indexing pass: incremental skipping, failure isolation, and stored data.

The vector store is a real on-disk Chroma instance; only the two network seams
(download, embed) are replaced with local fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from zotero_rag_core.indexer import chunk_id, index_plan
from zotero_rag_core.planning import IndexPlan, PlannedAttachment
from zotero_rag_core.vectorstore import VectorStore

PdfFactory = Callable[[str, list[str]], Path]


def planned(key: str = "ATT1", title: str = "A Paper") -> PlannedAttachment:
    return PlannedAttachment(
        attachment_key=key,
        item_key="ITEM1",
        title=title,
        authors="Doe, Jane",
        year="2021",
        doi="10.1/xyz",
        collection_name="Writing",
        zotero_url="https://www.zotero.org/u/items/ITEM1",
    )


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic 3-dim vectors; no network, no model."""
    return [[float(len(text)), 1.0, 0.5] for text in texts]


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma")


def test_indexes_an_attachment_end_to_end(store: VectorStore, pdf_factory: PdfFactory) -> None:
    pdf = pdf_factory("doc.pdf", ["Page one text", "Page two text"])
    plan = IndexPlan(attachments=[planned()])

    stats = index_plan(
        plan,
        store=store,
        download=lambda key: pdf,
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert stats.indexed == 1
    assert stats.pages == 2
    assert stats.chunks == 2
    assert stats.failed == 0
    assert stats.embedding_tokens > 0
    assert store.count() == 2


def test_stored_metadata_matches_the_prd_schema(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    pdf = pdf_factory("doc.pdf", ["Only page"])
    index_plan(
        IndexPlan(attachments=[planned()]),
        store=store,
        download=lambda key: pdf,
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
    )

    stored = store.get_by_attachment("ATT1")
    assert [chunk.id for chunk in stored] == [chunk_id("ATT1", 1, 0)]
    metadata = stored[0].metadata
    assert metadata["item_key"] == "ITEM1"
    assert metadata["title"] == "A Paper"
    assert metadata["authors"] == "Doe, Jane"
    assert metadata["year"] == "2021"
    assert metadata["doi"] == "10.1/xyz"
    assert metadata["collection_name"] == "Writing"
    assert str(metadata["zotero_url"]).endswith("ITEM1")
    assert metadata["page"] == 1
    assert metadata["chunk_index"] == 0
    # The chunk text itself is stored, so query can build context from it.
    assert "Only page" in stored[0].text


def test_second_run_skips_what_is_already_indexed(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    """Acceptance criterion 6, and the PoC's whole incremental rule."""
    pdf = pdf_factory("doc.pdf", ["Some text"])
    plan = IndexPlan(attachments=[planned()])
    kwargs = {
        "store": store,
        "download": lambda key: pdf,
        "embed": fake_embed,
        "chunk_size": 1000,
        "chunk_overlap": 200,
    }

    first = index_plan(plan, **kwargs)  # type: ignore[arg-type]
    second = index_plan(plan, **kwargs)  # type: ignore[arg-type]

    assert first.indexed == 1
    assert second.indexed == 0
    assert second.already_indexed == 1
    assert store.count() == first.chunks  # nothing duplicated


def test_skipping_avoids_re_embedding(store: VectorStore, pdf_factory: PdfFactory) -> None:
    """The skip must happen before the embedding call, not after."""
    pdf = pdf_factory("doc.pdf", ["Some text"])
    calls: list[int] = []

    def counting_embed(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return fake_embed(texts)

    plan = IndexPlan(attachments=[planned()])
    for _ in range(2):
        index_plan(
            plan,
            store=store,
            download=lambda key: pdf,
            embed=counting_embed,
            chunk_size=1000,
            chunk_overlap=200,
        )
    assert len(calls) == 1


def test_a_failing_attachment_does_not_abort_the_run(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    good = pdf_factory("good.pdf", ["Fine content"])

    def download(key: str) -> Path:
        if key == "BAD":
            raise OSError("network exploded")
        return good

    plan = IndexPlan(attachments=[planned("BAD", "Broken"), planned("OK", "Fine")])
    stats = index_plan(
        plan,
        store=store,
        download=download,
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert stats.failed == 1
    assert stats.indexed == 1
    assert stats.failures[0].title == "Broken"
    assert "network exploded" in stats.failures[0].error


def test_unreadable_pdf_is_recorded_as_a_failure(store: VectorStore, tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    stats = index_plan(
        IndexPlan(attachments=[planned()]),
        store=store,
        download=lambda key: broken,
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
    )
    assert stats.failed == 1
    assert stats.indexed == 0


def test_pdf_without_text_is_counted_separately_from_failures(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    scan = pdf_factory("scan.pdf", ["", " "])
    stats = index_plan(
        IndexPlan(attachments=[planned()]),
        store=store,
        download=lambda key: scan,
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
    )
    assert stats.no_text == 1
    assert stats.failed == 0
    assert stats.indexed == 0
    assert store.count() == 0


def test_embedding_count_mismatch_is_caught(store: VectorStore, pdf_factory: PdfFactory) -> None:
    """A short vector list would silently misalign chunks and metadata."""
    pdf = pdf_factory("doc.pdf", ["a" * 400])
    stats = index_plan(
        IndexPlan(attachments=[planned()]),
        store=store,
        download=lambda key: pdf,
        embed=lambda texts: [[0.0, 0.0, 0.0]],
        chunk_size=100,
        chunk_overlap=0,
    )
    assert stats.failed == 1
    assert store.count() == 0


def test_progress_hook_fires_once_per_attachment(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    pdf = pdf_factory("doc.pdf", ["text"])
    seen: list[str] = []
    plan = IndexPlan(attachments=[planned("A"), planned("B")])
    index_plan(
        plan,
        store=store,
        download=lambda key: pdf,
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
        on_start=lambda attachment: seen.append(attachment.attachment_key),
    )
    assert seen == ["A", "B"]


def test_empty_plan_is_a_no_op(store: VectorStore) -> None:
    stats = index_plan(
        IndexPlan(),
        store=store,
        download=lambda key: Path("/nonexistent"),
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
    )
    assert stats.planned == 0
    assert stats.indexed == 0
    assert store.count() == 0


def test_a_rejected_key_is_reported_without_the_unexpected_prefix(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    from zotero_rag_core.errors import RejectedCredentialError

    pdf = pdf_factory("doc.pdf", ["Some content"])

    def embed(texts: list[str]) -> list[list[float]]:
        raise RejectedCredentialError(
            "OpenRouter", "OPENROUTER_API_KEY", "HTTP 401: User not found."
        )

    stats = index_plan(
        IndexPlan(attachments=[planned()]),
        store=store,
        download=lambda key: pdf,
        embed=embed,
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert stats.failed == 1
    assert stats.failures[0].error.startswith(
        "OpenRouter rejected the API key in OPENROUTER_API_KEY"
    )


def test_parallel_indexing_matches_a_serial_run(tmp_path: Path, pdf_factory: PdfFactory) -> None:
    pdfs = {f"ATT{i}": pdf_factory(f"doc{i}.pdf", [f"Page {i} text", "More"]) for i in range(6)}
    plan = IndexPlan(attachments=[planned(key, f"Paper {key}") for key in pdfs])

    def run(workers: int, name: str) -> tuple[object, set[str]]:
        store = VectorStore(tmp_path / name)
        stats = index_plan(
            plan,
            store=store,
            download=pdfs.__getitem__,
            embed=fake_embed,
            chunk_size=1000,
            chunk_overlap=200,
            workers=workers,
        )
        return stats.model_dump(exclude={"elapsed_s"}), store.attachment_keys()

    assert run(4, "parallel") == run(1, "serial")


def test_a_failing_attachment_does_not_stop_the_other_workers(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    good = pdf_factory("good.pdf", ["Fine content"])

    def download(key: str) -> Path:
        if key == "BAD":
            raise OSError("network exploded")
        return good

    plan = IndexPlan(attachments=[planned("BAD", "Broken"), planned("OK1"), planned("OK2")])
    stats = index_plan(
        plan,
        store=store,
        download=download,
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
        workers=3,
    )

    assert (stats.indexed, stats.failed) == (2, 1)
    assert stats.failures[0].title == "Broken"


def test_the_store_is_written_only_from_the_calling_thread(
    store: VectorStore, pdf_factory: PdfFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    pdf = pdf_factory("doc.pdf", ["Content"])
    writers: set[str] = set()
    add = store.add

    def recording_add(**kwargs: object) -> None:
        writers.add(threading.current_thread().name)
        add(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "add", recording_add)
    index_plan(
        IndexPlan(attachments=[planned(f"ATT{i}") for i in range(4)]),
        store=store,
        download=lambda key: pdf,
        embed=fake_embed,
        chunk_size=1000,
        chunk_overlap=200,
        workers=4,
    )

    assert writers == {threading.current_thread().name}


def test_an_exception_from_the_progress_hook_stops_the_run(
    store: VectorStore, pdf_factory: PdfFactory
) -> None:
    class Stop(Exception):
        pass

    def on_start(attachment: PlannedAttachment) -> None:
        raise Stop

    pdf = pdf_factory("doc.pdf", ["Content"])
    with pytest.raises(Stop):
        index_plan(
            IndexPlan(attachments=[planned("ATT1"), planned("ATT2")]),
            store=store,
            download=lambda key: pdf,
            embed=fake_embed,
            chunk_size=1000,
            chunk_overlap=200,
            on_start=on_start,
            workers=2,
        )
