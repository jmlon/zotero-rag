"""`--store postgres://...`: running the CLI against a shared PostgreSQL store.

This is how the hosted retrieval stack is measured with the CLI's own `eval` and
question sets: the index folder still supplies `config.yaml` and the golden set,
and only the chunks live in the database.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_cli import (
    ONE_QUESTION,
    _eval_index_folder,
    _fake_library,
    _paper_with_note,
    _stub_seams,
)
from zotero_rag import cli
from zotero_rag.index_folder import STORE_ENV, build_keyword_index, open_index, open_store
from zotero_rag_core.errors import ConfigError, NotIndexedError
from zotero_rag_core.retrieval import BM25Searcher
from zotero_rag_core.stores.chroma import ChromaStore
from zotero_rag_core.stores.pg import PgFullTextSearcher, PgStore

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.delenv(STORE_ENV, raising=False)
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def folder_on_postgres(tmp_path: Path, pg_store: PgStore, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The CLI tests' one-chunk index folder, with its chunks moved into Postgres.

    The folder's own Chroma store is removed afterwards, so nothing can be
    answered from it by mistake.
    """
    _eval_index_folder(tmp_path, ONE_QUESTION)
    chroma = ChromaStore(tmp_path / "chroma")
    chunks = chroma.all_chunks()
    pg_store.add(
        ids=[chunk.id for chunk in chunks],
        embeddings=[[1.0, 0.0, 0.5]],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )
    fingerprint = chroma.read_fingerprint()
    assert fingerprint is not None
    pg_store.write_fingerprint(fingerprint)
    shutil.rmtree(tmp_path / "chroma")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_store_takes_only_a_postgres_url(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="PostgreSQL URL"):
        open_store(tmp_path, "sqlite:///index.db", create=False)


def test_without_store_the_folder_keeps_its_own(tmp_path: Path) -> None:
    assert isinstance(open_store(tmp_path, None, create=False), ChromaStore)


def test_a_database_searches_its_own_full_text_index(pg_store: PgStore, tmp_path: Path) -> None:
    assert isinstance(build_keyword_index(pg_store, "hybrid"), PgFullTextSearcher)
    assert build_keyword_index(pg_store, "vector") is None
    assert isinstance(build_keyword_index(ChromaStore(tmp_path / "c"), "hybrid"), BM25Searcher)


def test_query_answers_from_the_database(
    folder_on_postgres: Path, pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_seams(monkeypatch)
    result = runner.invoke(cli.app, ["query", "--store", pg_url, "--json", "What is a fog node?"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [source["item_key"] for source in payload["sources"]] == ["ITEM1"]


def test_the_store_can_come_from_the_environment(
    folder_on_postgres: Path, pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_seams(monkeypatch)
    monkeypatch.setenv(STORE_ENV, pg_url)
    result = runner.invoke(cli.app, ["query", "--json", "What is a fog node?"])
    assert result.exit_code == 0, result.output


def test_eval_grades_the_database(
    folder_on_postgres: Path, pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parity measurement zotero-rag-web relies on."""
    _stub_seams(monkeypatch)
    result = runner.invoke(cli.app, ["eval", "--store", pg_url, "--retrieval", "hybrid", "--json"])
    assert result.exit_code == 0, result.output
    (outcome,) = json.loads(result.stdout)["results"]
    assert outcome["hit"] is True and outcome["chunks_retrieved"] == 1


def test_an_empty_database_says_to_index_first(
    pg_store: PgStore, pg_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text("index:\n  top_k: 3\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(NotIndexedError, match="index --store"):
        open_index(lambda notice: None, pg_url)


def test_help_never_shows_the_url_from_the_environment(
    pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(STORE_ENV, "postgresql://user:s3cret@db.example/zr")
    result = runner.invoke(cli.app, ["query", "--help"])
    assert "--store" in result.output
    assert "s3cret" not in result.output


def test_index_writes_to_the_database_and_reruns_incrementally(
    tmp_path: Path, pg_store: PgStore, pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole `index` run into Postgres: chunks, membership, fingerprint, no re-embedding."""
    (tmp_path / "config.yaml").write_text("index:\n  collection_key: NET\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    paper = _paper_with_note("P1")
    _fake_library(monkeypatch, {"ROUTING": paper, "QUIC": _paper_with_note("P2")})

    first = runner.invoke(cli.app, ["index", "--store", pg_url])
    assert first.exit_code == 0, first.output
    assert not (tmp_path / "chroma").exists()  # nothing went to the folder's own store
    assert pg_store.collection_keys_by_attachment() == {"NP1": ["ROUTING"], "NP2": ["QUIC"]}
    fingerprint = pg_store.read_fingerprint()
    assert fingerprint is not None and fingerprint.embedding_dim == 3

    # P1 moves to QUIC: membership changes, nothing is re-embedded.
    _fake_library(monkeypatch, {"QUIC": [*paper, *_paper_with_note("P2")]})
    second = runner.invoke(cli.app, ["index", "--store", pg_url])
    assert second.exit_code == 0, second.output
    assert "Updated the folder membership of 1 attachment (no re-embedding)" in second.output
    assert pg_store.collection_keys_by_attachment() == {"NP1": ["QUIC"], "NP2": ["QUIC"]}


def test_index_leaves_the_keyword_statistics_fresh(
    tmp_path: Path, pg_store: PgStore, pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """So the first question after indexing does not wait for the refresh."""
    (tmp_path / "config.yaml").write_text("index:\n  collection_key: NET\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _fake_library(monkeypatch, {"ROUTING": _paper_with_note("P1")})
    assert runner.invoke(cli.app, ["index", "--store", pg_url]).exit_code == 0
    assert not pg_store.keyword_stats_stale()
