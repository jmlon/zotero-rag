"""What is particular to the PostgreSQL store: SQL filters and full-text search.

The behaviour every store shares is in test_store_contract.py. Here the SQL
translation of `Filters` is held to the same rule as the Chroma one, over the same
grid of conditions (test_filters.py), and full-text search is held to BM25's
definition of a match.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_filters import LIBRARY, chunk_metadata, every_filter, expected, found
from zotero_rag_core.retrieval import BM25Searcher, KeywordSearcher
from zotero_rag_core.store import ChunkStore, Filters
from zotero_rag_core.stores.pg import PgFullTextSearcher, PgStore, filters_sql, tsvector_text


@pytest.fixture
def library(pg_store: PgStore) -> PgStore:
    pg_store.add(
        ids=[f"{key}:1:0" for key in LIBRARY],
        embeddings=[[1.0, float(position), 0.0] for position in range(len(LIBRARY))],
        documents=[f"Retrieval paper {key}." for key in LIBRARY],
        metadatas=[chunk_metadata(key) for key in LIBRARY],
    )
    return pg_store


def test_sql_filters_agree_with_the_rule_on_every_combination(library: PgStore) -> None:
    """One database, every filter: the grid is 96 combinations."""
    searcher = PgFullTextSearcher(library)
    for filters in every_filter():
        want = expected(filters)
        assert found(library.query([1.0, 0.0, 0.0], len(LIBRARY), filters)) == want, filters
        assert found(searcher.search("retrieval", len(LIBRARY), filters)) == want, filters


def test_no_filter_means_no_where_clause() -> None:
    assert filters_sql(None) == ("", {})
    assert filters_sql(Filters()) == ("", {})


def test_filter_values_are_parameters_not_sql() -> None:
    clause, params = filters_sql(Filters(tags=frozenset({"x'); DROP TABLE chunks; --"})))
    assert "DROP" not in clause
    assert params == {"tags": ["x'); DROP TABLE chunks; --"]}


# --- full-text search matches what BM25 matches -----------------------------------------

PASSAGES = {
    "acronym": "CFR-RL schedules flows with reinforcement learning.",
    "accent": "La utilización del enlace sube con ECMP.",
    "version": "Results for GPT 3.5 and other models.",
    "plain": "A survey of congestion control.",
    "email": "Write to the authors by e-mail.",
}

QUESTIONS = [
    "What is CFR-RL?",
    "cfr rl",
    "utilizacion del enlace",
    "¿Qué dice sobre la utilización?",
    "GPT 3.5",
    "e-mail",
    "congestion",
    "nothing in common",
    "!!!",
]


@pytest.fixture
def passages(pg_store: PgStore) -> PgStore:
    keys = list(PASSAGES)
    pg_store.add(
        ids=[f"{key}:1:0" for key in keys],
        embeddings=[[1.0, 0.0, 0.0] for _ in keys],
        documents=list(PASSAGES.values()),
        metadatas=[{"attachment_key": key, "page": 1, "chunk_index": 0} for key in keys],
    )
    return pg_store


@pytest.mark.parametrize("question", QUESTIONS)
def test_full_text_search_matches_the_chunks_bm25_matches(passages: PgStore, question: str) -> None:
    """Same tokens on both sides, so the same chunks match; only the ranking differs."""
    bm25 = BM25Searcher(passages.all_chunks())
    pg = PgFullTextSearcher(passages)
    assert found(pg.search(question, 10)) == found(bm25.search(question, 10))


def test_the_tsvector_is_built_from_bm25s_tokens() -> None:
    """Postgres' parser keeps `3.5` and accents; `tokenize` splits and folds them."""
    assert tsvector_text("GPT 3.5 utilización CFR-RL") == "gpt 3 5 utilizacion cfr rl"


def test_a_better_match_ranks_first(pg_store: PgStore) -> None:
    texts = {"A": "ECMP once.", "B": "ECMP ECMP ECMP and ECMP hashing with ECMP."}
    pg_store.add(
        ids=["A:1:0", "B:1:0"],
        embeddings=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        documents=list(texts.values()),
        metadatas=[{"attachment_key": key, "page": 1, "chunk_index": 0} for key in texts],
    )
    hits = PgFullTextSearcher(pg_store).search("ECMP hashing", 2)
    assert [hit.id for hit in hits] == ["B:1:0", "A:1:0"]


def test_keyword_search_respects_the_limit_and_empty_filters(passages: PgStore) -> None:
    searcher = PgFullTextSearcher(passages)
    assert len(searcher.search("the", 1)) == 1
    assert searcher.search("the", 0) == []
    assert searcher.search("the", 5, Filters(tags=frozenset())) == []


# --- what the database holds ------------------------------------------------------------


def test_an_item_is_one_row_however_many_chunks_it_has(pg_store: PgStore, pg_url: str) -> None:
    import psycopg

    metadata: dict[str, Any] = {"attachment_key": "A", "item_key": "I", "title": "T"}
    pg_store.add(
        ids=["A:1:0", "A:1:1", "A:2:0"],
        embeddings=[[1.0, 0.0, 0.0]] * 3,
        documents=["a", "b", "c"],
        metadatas=[metadata | {"page": 1}, metadata | {"page": 1}, metadata | {"page": 2}],
    )
    with psycopg.connect(pg_url) as conn:
        assert conn.execute("SELECT count(*) FROM items").fetchone() == (1,)
        assert conn.execute("SELECT count(*) FROM attachments").fetchone() == (1,)


def test_the_vector_index_is_used_for_the_query(pg_store: PgStore, pg_url: str) -> None:
    """The query's cast must match the index expression, or every search is a scan."""
    import psycopg

    pg_store.add(
        ids=["A:1:0"],
        embeddings=[[1.0, 0.0, 0.0]],
        documents=["a"],
        metadatas=[{"attachment_key": "A"}],
    )
    with psycopg.connect(pg_url) as conn:
        conn.execute("SET enable_seqscan = off")
        plan = conn.execute(
            "EXPLAIN SELECT id FROM chunks "
            "ORDER BY embedding::vector(3) <=> '{1,0,0}'::float8[]::vector(3) LIMIT 5"
        ).fetchall()
    assert any("chunks_embedding_hnsw" in line for (line,) in plan)


#: Planner settings that leave the HNSW index scan as the only reasonable plan.
FORCE_HNSW = ("enable_seqscan", "enable_bitmapscan", "enable_sort")


def test_a_narrow_filter_still_fills_top_k_through_the_index(
    pg_store: PgStore, pg_url: str
) -> None:
    """Without iterative scans, HNSW would hand back its ~100 nearest rows and the
    filter would then discard them all: a small folder far from the question would
    come back empty. The HNSW path is forced here: on a table this small the planner
    would rather find the folder's rows through the GIN index and sort them exactly.
    """
    import psycopg

    near = [[1.0, 0.001 * n, 0.0] for n in range(300)]
    # Distinct distances: exact ties are where an approximate index may drop a row.
    far = [[0.3 - 0.02 * n, 0.0, 1.0] for n in range(10)]
    pg_store.add(
        ids=[f"N{n}:1:0" for n in range(300)] + [f"F{n}:1:0" for n in range(10)],
        embeddings=near + far,
        documents=["near"] * 300 + ["far"] * 10,
        metadatas=[{"attachment_key": f"N{n}", "collection_keys": ["BIG"]} for n in range(300)]
        + [{"attachment_key": f"F{n}", "collection_keys": ["SMALL"]} for n in range(10)],
    )
    with psycopg.connect(pg_url, autocommit=True) as conn:
        database = conn.execute("SELECT current_database()").fetchone()[0]  # type: ignore[index]
        for setting in FORCE_HNSW:
            conn.execute(f'ALTER DATABASE "{database}" SET {setting} = off')
    forced = PgStore.from_url(pg_url)
    try:
        hits = forced.query([1.0, 0.0, 0.0], 10, Filters(collection_keys=frozenset({"SMALL"})))
    finally:
        forced.close()
        with psycopg.connect(pg_url, autocommit=True) as conn:
            for setting in FORCE_HNSW:
                conn.execute(f'ALTER DATABASE "{database}" RESET {setting}')
    assert len(hits) == 10 and {hit.text for hit in hits} == {"far"}


def test_the_store_and_searcher_satisfy_the_protocols(pg_store: PgStore) -> None:
    """Checked by mypy through these annotations."""
    store: ChunkStore = pg_store
    searcher: KeywordSearcher = PgFullTextSearcher(pg_store)
    assert store.count() == 0 and searcher.search("anything", 5) == []
