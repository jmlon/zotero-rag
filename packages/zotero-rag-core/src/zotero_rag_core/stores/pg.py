"""A shared PostgreSQL store with pgvector: a `ChunkStore`, and its keyword search.

What Chroma keeps as one flat metadata mapping per chunk is normalised here into
items, attachments and chunks (`pg_schema.sql`), so that a change to an item, such
as being filed in another folder, is one row, and so that filters are plain
column conditions. Reading a chunk back joins the three and rebuilds the same flat
mapping, so every consumer of `ChunkStore` sees what it would see from Chroma.

Two deliberate differences from `ChromaStore`:

- Distances are cosine distances. Chroma's default, used by index folders, is
  squared L2. For the unit-length vectors embedding models return the two rank
  identically, but their values are not comparable across stores.
- Folder membership belongs to the item, so `set_collection_keys` on one
  attachment moves its sibling attachments too. Zotero files items, not files,
  so that is the right model; Chroma merely cannot express it.

Requires the `pgvector` extra (psycopg 3) and the `vector` extension in the database.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from typing import Any, cast

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from zotero_rag_core.fingerprint import StoreFingerprint
from zotero_rag_core.retrieval import SCORE_DECIMALS, tokenize
from zotero_rag_core.store import Filters, StoredChunk, VectorStoreError, reading_order_key

logger = logging.getLogger(__name__)

SCHEMA_RESOURCE = "pg_schema.sql"
VECTOR_INDEX = "chunks_embedding_hnsw"
#: HNSW candidates examined per query; raised to `top_k` when that is larger.
EF_SEARCH = 100
#: pgvector's HNSW index takes at most this many dimensions for `vector`.
MAX_INDEXED_DIMENSIONS = 2000
#: rank_bm25.BM25Okapi's defaults, which `BM25Searcher` uses.
BM25_K1, BM25_B, BM25_EPSILON = 1.5, 0.75, 0.25
#: Marks the keyword statistics stale; run inside every transaction that writes chunks.
_MARK_STALE = "UPDATE chunk_store SET keyword_stale = true WHERE id = 1"

#: Chunk metadata keys stored on the item row; NULL when the metadata omits them.
ITEM_FIELDS = (
    "item_type",
    "title",
    "authors",
    "year",
    "year_int",
    "doi",
    "zotero_url",
    "collection_keys",
    "tags",
)
#: Chunk metadata key → attachment column.
ATTACHMENT_FIELDS = {
    "attachment_version": "version",
    "document_format": "document_format",
    "collection_name": "collection_name",
}
#: Chunk metadata keys stored on the chunk row.
CHUNK_FIELDS = ("page", "chunk_index", "locator_kind", "locator_label", "embedding_model")
_COLUMNS = {"item_key", "attachment_key", *ITEM_FIELDS, *ATTACHMENT_FIELDS, *CHUNK_FIELDS}

_SELECT = """
SELECT c.id, c.text, c.attachment_key, a.item_key, c.extra,
       c.page, c.chunk_index, c.locator_kind, c.locator_label, c.embedding_model,
       a.version AS attachment_version, a.document_format, a.collection_name,
       i.item_type, i.title, i.authors, i.year, i.year_int, i.doi, i.zotero_url,
       i.collection_keys, i.tags
FROM chunks c
JOIN attachments a ON a.key = c.attachment_key
JOIN items i ON i.key = a.item_key
"""


def schema_sql() -> str:
    """The DDL `PgStore` expects, as shipped with this package."""
    return (resources.files(__package__) / SCHEMA_RESOURCE).read_text(encoding="utf-8")


def filters_sql(filters: Filters | None) -> tuple[str, dict[str, Any]]:
    """A `WHERE` clause over the item row (`i`) for `filters`, and its parameters.

    The same rule as `Filters.matches`: a NULL list or year fails any condition on
    it, `year_int` 0 is an undated item, and dated items are bounded above 0 so an
    open-ended "up to 2018" cannot let undated items through.
    """
    if filters is None:
        return "", {}
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if filters.collection_keys is not None:
        clauses.append("i.collection_keys && %(collection_keys)s::text[]")
        params["collection_keys"] = sorted(filters.collection_keys)
    if filters.tags is not None:
        clauses.append("i.tags && %(tags)s::text[]")
        params["tags"] = sorted(filters.tags)
    if filters.limits_year:
        dated = ["i.year_int > 0"]
        if filters.year_min is not None:
            dated.append("i.year_int >= %(year_min)s")
            params["year_min"] = filters.year_min
        if filters.year_max is not None:
            dated.append("i.year_int <= %(year_max)s")
            params["year_max"] = filters.year_max
        condition = "(" + " AND ".join(dated) + ")"
        if filters.include_undated:
            condition = f"({condition} OR i.year_int = 0)"
        clauses.append(condition)
    if not clauses:
        return "", {}
    return "WHERE " + " AND ".join(clauses), params


def tsvector_text(text: str) -> str:
    """The text `chunks.tsv` is built from: its tokens as BM25 sees them.

    Postgres' own parser keeps `3.5`, `e-mail` and accents whole, where
    `retrieval.tokenize` splits and folds them; feeding it the already-split
    tokens makes both keyword searches match the same chunks.
    """
    return " ".join(tokenize(text))


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    """The flat chunk metadata, as it was written, from a joined row."""
    metadata: dict[str, Any] = {
        "item_key": row["item_key"],
        "attachment_key": row["attachment_key"],
    }
    for field in (*ITEM_FIELDS, *ATTACHMENT_FIELDS, *CHUNK_FIELDS):
        value = row[field]
        if value is not None:
            metadata[field] = list(value) if isinstance(value, list) else value
    metadata.update(row["extra"] or {})
    return metadata


def _chunk(row: dict[str, Any]) -> StoredChunk:
    distance = row.get("distance")
    return StoredChunk(
        id=row["id"],
        text=row["text"],
        metadata=_metadata(row),
        distance=float(distance) if distance is not None else None,
    )


def without_nul(value: Any) -> Any:
    """`value` with every NUL character removed from its strings, however nested.

    PDF text extraction sometimes leaves NULs (often from text stored as UTF-16), and
    PostgreSQL refuses them in text and JSON columns. Chroma keeps them; removing them
    here leaves the CLI's Chroma indexes as they are.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {without_nul(key): without_nul(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(without_nul(item) for item in value)
    return value


class PgStore:
    """Chunks in PostgreSQL, searched with pgvector. Thread-safe through its pool."""

    def __init__(self, pool: ConnectionPool[psycopg.Connection[Any]]) -> None:
        self._pool = pool
        self._indexed_dimension: int | None = None

    @classmethod
    def from_url(cls, url: str, *, create: bool = False, max_size: int = 4) -> PgStore:
        """A store on its own connection pool. `create` runs `pg_schema.sql` first."""
        try:
            pool: ConnectionPool[psycopg.Connection[Any]] = ConnectionPool(
                url, min_size=1, max_size=max_size, open=True
            )
            pool.wait(timeout=30)
        except (psycopg.Error, OSError) as exc:
            raise VectorStoreError(f"Cannot connect to the database: {exc}") from exc
        store = cls(pool)
        if create:
            store.create_schema()
        return store

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def _cursor(self, action: str) -> Iterator[psycopg.Cursor[dict[str, Any]]]:
        """A cursor in one transaction; any database error becomes `VectorStoreError`."""
        try:
            with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
                yield cur
        except psycopg.Error as exc:
            raise VectorStoreError(f"Cannot {action}: {exc}") from exc

    def create_schema(self) -> None:
        with self._cursor("create the store's tables") as cur:
            cur.execute(cast(Any, schema_sql()))

    # --- writing -----------------------------------------------------------

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store new chunks. An id already stored is left as it is, as Chroma does."""
        self._write(ids, embeddings, documents, metadatas, replace=False)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._write(ids, embeddings, documents, metadatas, replace=True)

    def _write(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        *,
        replace: bool,
    ) -> None:
        if not ids:
            return
        if not len(ids) == len(embeddings) == len(documents) == len(metadatas):
            raise ValueError("ids, embeddings, documents and metadatas differ in length")

        items: dict[str, dict[str, Any]] = {}
        attachments: dict[str, dict[str, Any]] = {}
        chunks: list[dict[str, Any]] = []
        for identifier, embedding, text, metadata in zip(
            ids, embeddings, documents, metadatas, strict=True
        ):
            text, metadata = without_nul(text), without_nul(metadata)
            attachment_key = str(metadata.get("attachment_key") or "")
            if not attachment_key:
                raise ValueError(f"chunk {identifier!r} has no attachment_key")
            # Without an item, the attachment stands for itself.
            item_key = str(metadata.get("item_key") or attachment_key)
            items[item_key] = {"key": item_key} | {f: metadata.get(f) for f in ITEM_FIELDS}
            attachments[attachment_key] = {"key": attachment_key, "item_key": item_key} | {
                column: metadata.get(field) for field, column in ATTACHMENT_FIELDS.items()
            }
            tokens = tokenize(text)
            chunks.append(
                {"id": identifier, "attachment_key": attachment_key, "text": text}
                | {field: metadata.get(field) for field in CHUNK_FIELDS}
                | {
                    "embedding": [float(value) for value in embedding],
                    "tsv": " ".join(tokens),
                    # BM25's document length, and whether BM25Searcher would skip it.
                    "n_tokens": len(tokens),
                    "blank": not text.strip(),
                    "extra": Jsonb({k: v for k, v in metadata.items() if k not in _COLUMNS}),
                }
            )

        chunk_columns = ", ".join(CHUNK_FIELDS)
        chunk_values = ", ".join(f"%({field})s" for field in CHUNK_FIELDS)
        on_conflict = (
            "DO UPDATE SET attachment_key = EXCLUDED.attachment_key, text = EXCLUDED.text, "
            "embedding = EXCLUDED.embedding, tsv = EXCLUDED.tsv, extra = EXCLUDED.extra, "
            "n_tokens = EXCLUDED.n_tokens, blank = EXCLUDED.blank, "
            + ", ".join(f"{field} = EXCLUDED.{field}" for field in CHUNK_FIELDS)
            if replace
            else "DO NOTHING"
        )
        dimension = len(chunks[0]["embedding"])
        if dimension > MAX_INDEXED_DIMENSIONS:
            raise VectorStoreError(
                f"These vectors have {dimension} dimensions, and pgvector can index at most "
                f"{MAX_INDEXED_DIMENSIONS}. Choose an embedding model with "
                f"{MAX_INDEXED_DIMENSIONS} dimensions or fewer for a PostgreSQL store "
                "(text-embedding-3-small has 1536)."
            )
        with self._cursor("write to the store") as cur:
            cur.executemany(
                cast(
                    Any,
                    f"INSERT INTO items (key, {', '.join(ITEM_FIELDS)}) "
                    f"VALUES (%(key)s, {', '.join(f'%({f})s' for f in ITEM_FIELDS)}) "
                    "ON CONFLICT (key) DO UPDATE SET "
                    + ", ".join(f"{field} = EXCLUDED.{field}" for field in ITEM_FIELDS),
                ),
                list(items.values()),
            )
            cur.executemany(
                "INSERT INTO attachments (key, item_key, version, document_format, "
                "collection_name) VALUES (%(key)s, %(item_key)s, %(version)s, "
                "%(document_format)s, %(collection_name)s) ON CONFLICT (key) DO UPDATE SET "
                "item_key = EXCLUDED.item_key, version = EXCLUDED.version, "
                "document_format = EXCLUDED.document_format, "
                "collection_name = EXCLUDED.collection_name",
                list(attachments.values()),
            )
            cur.executemany(
                cast(
                    Any,
                    f"INSERT INTO chunks (id, attachment_key, text, {chunk_columns}, "
                    "embedding, tsv, n_tokens, blank, extra) "
                    "VALUES (%(id)s, %(attachment_key)s, %(text)s, "
                    f"{chunk_values}, %(embedding)s::float8[]::vector, "
                    "to_tsvector('simple', %(tsv)s), %(n_tokens)s, %(blank)s, %(extra)s) "
                    f"ON CONFLICT (id) {on_conflict}",
                ),
                chunks,
            )
            cur.execute(_MARK_STALE)
            self._ensure_vector_index(cur, dimension)

    def _ensure_vector_index(self, cur: psycopg.Cursor[dict[str, Any]], dimension: int) -> None:
        """Create the HNSW index for vectors of `dimension`, once that width is known.

        The index casts to a fixed width, so it also refuses vectors of any other
        width: mixing two models' vectors fails loudly instead of searching nonsense.
        """
        if self._indexed_dimension == dimension:
            return
        cur.execute(
            cast(
                Any,
                f"CREATE INDEX IF NOT EXISTS {VECTOR_INDEX} ON chunks USING hnsw "
                f"((embedding::vector({int(dimension)})) vector_cosine_ops)",
            )
        )
        self._indexed_dimension = dimension

    def delete_by_attachment(self, attachment_key: str) -> int:
        with self._cursor("delete from the store") as cur:
            cur.execute("DELETE FROM chunks WHERE attachment_key = %s", (attachment_key,))
            removed = cur.rowcount
            cur.execute("DELETE FROM attachments WHERE key = %s", (attachment_key,))
            cur.execute(_MARK_STALE)
        return max(removed, 0)

    def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._cursor("delete from the store") as cur:
            cur.execute("DELETE FROM chunks WHERE id = ANY(%s::text[])", (list(ids),))
            cur.execute(_MARK_STALE)

    def set_collection_keys(self, attachment_key: str, collection_keys: list[str]) -> int:
        """File the attachment's item in `collection_keys`; returns the attachment's chunks.

        Membership is the item's, so its other attachments move with it.
        """
        if not collection_keys:
            raise ValueError("Give at least one collection key; an empty list selects nothing")
        with self._cursor("update the store") as cur:
            cur.execute(
                "SELECT count(*) AS n FROM chunks WHERE attachment_key = %s", (attachment_key,)
            )
            row = cur.fetchone()
            count = int(row["n"]) if row else 0
            if count:
                cur.execute(
                    "UPDATE items SET collection_keys = %s::text[] "
                    "WHERE key = (SELECT item_key FROM attachments WHERE key = %s)",
                    (list(collection_keys), attachment_key),
                )
        return count

    def reset(self) -> None:
        """Remove every chunk and attachment, and the vector index; items stay.

        Item rows hold bibliographic facts, not vectors, so a change of embedding
        model has no reason to forget them.
        """
        with self._cursor("reset the store") as cur:
            cur.execute("TRUNCATE chunks, attachments")
            cur.execute(cast(Any, f"DROP INDEX IF EXISTS {VECTOR_INDEX}"))
            cur.execute("UPDATE chunk_store SET fingerprint = '{}'::jsonb")
            cur.execute(_MARK_STALE)
        self._indexed_dimension = None

    # --- keyword statistics -------------------------------------------------

    def keyword_stats_stale(self) -> bool:
        return bool(self._scalar("SELECT keyword_stale FROM chunk_store WHERE id = 1"))

    def refresh_keyword_stats(self) -> None:
        """Rebuild what BM25 ranking needs: per-term document counts, and corpus totals.

        Call it once after a batch of writes; a search refreshes stale statistics
        itself, but then that search pays for it. It runs in one transaction, so
        searches keep the previous statistics until it commits.

        The corpus is BM25Searcher's: chunks that are not blank. The idf floor is
        rank_bm25's: a negative idf becomes `BM25_EPSILON` × the vocabulary's mean idf.
        """
        with self._cursor("refresh the keyword statistics") as cur:
            # Chunks written before these columns existed get them from their tsvector.
            cur.execute(
                "UPDATE chunks SET n_tokens = (SELECT coalesce(sum(cardinality(positions)), 0) "
                "FROM unnest(tsv)), blank = (btrim(text, E' \\t\\n\\r') = '') "
                "WHERE n_tokens IS NULL OR blank IS NULL"
            )
            cur.execute("DELETE FROM term_df")
            cur.execute(
                "INSERT INTO term_df (word, ndoc) "
                "SELECT word, ndoc FROM ts_stat('SELECT tsv FROM chunks WHERE NOT blank')"
            )
            cur.execute(
                "SELECT count(*) AS docs, coalesce(avg(n_tokens), 0) AS avgdl "
                "FROM chunks WHERE NOT blank"
            )
            totals = cur.fetchone() or {"docs": 0, "avgdl": 0}
            docs = int(totals["docs"])
            cur.execute(
                "SELECT coalesce(avg(ln((%s - ndoc + 0.5) / (ndoc + 0.5))), 0) AS mean_idf "
                "FROM term_df",
                (float(docs),),
            )
            row = cur.fetchone()
            mean_idf = float(row["mean_idf"]) if row else 0.0
            cur.execute(
                "UPDATE chunk_store SET keyword_stale = false, keyword_docs = %s, "
                "keyword_avgdl = %s, keyword_epsilon_idf = %s WHERE id = 1",
                (docs, float(totals["avgdl"]), BM25_EPSILON * mean_idf),
            )
        logger.info("keyword statistics refreshed over %d chunk(s)", docs)

    # --- reading -----------------------------------------------------------

    def query(
        self, embedding: list[float], top_k: int, filters: Filters | None = None
    ) -> list[StoredChunk]:
        """The `top_k` nearest chunks by cosine distance that pass `filters`.

        Filtering happens inside the index scan: pgvector's iterative scan keeps
        going until `top_k` matching rows are found, so a narrow folder still fills
        its quota. The relaxed order it allows is restored here by sorting.
        """
        if top_k <= 0 or (filters is not None and filters.selects_nothing):
            return []
        dimension = int(len(embedding))
        where, params = filters_sql(filters)
        # The cast must be written exactly as in the index expression for the index
        # to be used.
        distance = f"c.embedding::vector({dimension}) <=> %(q)s::float8[]::vector({dimension})"
        select = _SELECT.replace("FROM chunks c", f"     , {distance} AS distance\nFROM chunks c")
        statement = f"{select} {where} ORDER BY {distance} LIMIT %(k)s"
        params |= {"q": [float(value) for value in embedding], "k": top_k}
        with self._cursor("query the store") as cur:
            cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            cur.execute(cast(Any, f"SET LOCAL hnsw.ef_search = {max(EF_SEARCH, int(top_k))}"))
            cur.execute(cast(Any, statement), params)
            rows = cur.fetchall()
        chunks = [_chunk(row) for row in rows]
        chunks.sort(key=lambda chunk: (chunk.distance or 0.0, chunk.id))
        return chunks

    def _select(self, condition: str = "", params: Sequence[Any] = ()) -> list[StoredChunk]:
        with self._cursor("read the store") as cur:
            cur.execute(cast(Any, f"{_SELECT} {condition}"), params)
            return [_chunk(row) for row in cur.fetchall()]

    def get_by_attachment(self, attachment_key: str) -> list[StoredChunk]:
        chunks = self._select("WHERE c.attachment_key = %s", (attachment_key,))
        return sorted(chunks, key=reading_order_key)

    def get_by_ids(self, ids: list[str]) -> list[StoredChunk]:
        if not ids:
            return []
        return self._select("WHERE c.id = ANY(%s::text[])", (list(ids),))

    def all_chunks(self) -> list[StoredChunk]:
        return self._select("ORDER BY c.id")

    def _scalar(self, statement: str, params: Sequence[Any] = ()) -> Any:
        with self._cursor("read the store") as cur:
            cur.execute(cast(Any, statement), params)
            row = cur.fetchone()
        return None if row is None else next(iter(row.values()))

    def count(self) -> int:
        return int(self._scalar("SELECT count(*) FROM chunks"))

    def has_attachment(self, attachment_key: str) -> bool:
        return bool(
            self._scalar(
                "SELECT EXISTS (SELECT 1 FROM chunks WHERE attachment_key = %s)", (attachment_key,)
            )
        )

    def attachment_keys(self) -> set[str]:
        with self._cursor("read the store") as cur:
            cur.execute("SELECT DISTINCT attachment_key FROM chunks")
            return {str(row["attachment_key"]) for row in cur.fetchall()}

    def attachment_version(self, attachment_key: str) -> int | None:
        version = self._scalar(
            "SELECT a.version FROM attachments a WHERE a.key = %s "
            "AND EXISTS (SELECT 1 FROM chunks c WHERE c.attachment_key = a.key)",
            (attachment_key,),
        )
        return None if version is None else int(version)

    def collection_keys_by_attachment(self) -> dict[str, list[str] | None]:
        with self._cursor("read the store") as cur:
            cur.execute(
                "SELECT a.key, i.collection_keys FROM attachments a "
                "JOIN items i ON i.key = a.item_key "
                "WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.attachment_key = a.key)"
            )
            return {
                str(row["key"]): sorted(row["collection_keys"]) if row["collection_keys"] else None
                for row in cur.fetchall()
            }

    def stale_ids(self, embedding_model: str) -> list[str]:
        with self._cursor("read the store") as cur:
            cur.execute(
                "SELECT id FROM chunks WHERE embedding_model IS DISTINCT FROM %s ORDER BY id",
                (embedding_model,),
            )
            return [str(row["id"]) for row in cur.fetchall()]

    # --- how the store was built -------------------------------------------

    def read_fingerprint(self) -> StoreFingerprint | None:
        recorded = self._scalar("SELECT fingerprint FROM chunk_store WHERE id = 1")
        return StoreFingerprint.from_metadata(recorded or None)

    def write_fingerprint(self, fingerprint: StoreFingerprint) -> None:
        with self._cursor("record the store fingerprint") as cur:
            cur.execute(
                "UPDATE chunk_store SET fingerprint = fingerprint || %s WHERE id = 1",
                (Jsonb(fingerprint.to_metadata()),),
            )

    def derived_dimension(self) -> int | None:
        dimension = self._scalar("SELECT vector_dims(embedding) FROM chunks LIMIT 1")
        return None if dimension is None else int(dimension)


#: BM25Okapi in SQL. `matched` is every chunk sharing a token with the question
#: (within the filters); each matched (chunk, query term) pair scores
#: idf × tf × (k1 + 1) / (tf + k1 × (1 − b + b × length / mean length)). Query terms
#: count with their multiplicity, as rank_bm25 sums over the query's token list, and a
#: term the corpus lacks scores nothing.
_BM25_SQL = """
WITH q(term) AS (SELECT unnest(%(tokens)s::text[])),
idf AS (
    SELECT q.term,
           CASE WHEN raw.idf < 0 THEN s.keyword_epsilon_idf ELSE raw.idf END AS idf
    FROM q
    JOIN term_df d ON d.word = q.term
    CROSS JOIN chunk_store s
    CROSS JOIN LATERAL (
        SELECT ln((s.keyword_docs - d.ndoc + 0.5) / (d.ndoc + 0.5)) AS idf
    ) raw
),
matched AS (
    SELECT c.id, c.tsv, c.n_tokens
    FROM chunks c
    JOIN attachments a ON a.key = c.attachment_key
    JOIN items i ON i.key = a.item_key
    WHERE NOT c.blank AND c.tsv @@ to_tsquery('simple', %(query)s) {filters}
),
tf AS (
    SELECT m.id, m.n_tokens, u.lexeme, cardinality(u.positions) AS f
    FROM matched m, unnest(m.tsv) u
    WHERE u.lexeme = ANY(%(tokens)s::text[])
),
summed AS (
    SELECT tf.id,
           sum(idf.idf * tf.f * (%(k1)s + 1)
               / (tf.f + %(k1)s * (1 - %(b)s + %(b)s * tf.n_tokens
                  / nullif(s.keyword_avgdl, 0)))) AS score
    FROM tf
    JOIN idf ON idf.term = tf.lexeme
    CROSS JOIN chunk_store s
    GROUP BY tf.id
),
scored AS (
    SELECT id, score FROM summed
    ORDER BY round(score::numeric, %(decimals)s) DESC, id
    LIMIT %(limit)s
)
"""


class PgFullTextSearcher:
    """Keyword search in the database, ranked by BM25: a `retrieval.KeywordSearcher`.

    It returns what `BM25Searcher` returns over the same chunks, in the same order:
    the chunks sharing a token with the question (both sides tokenized by
    `retrieval.tokenize`), ranked by rank_bm25's BM25Okapi. Postgres' own
    `ts_rank_cd` was measured and rejected: without inverse document frequency,
    common words decided the order, and hybrid retrieval did worse with it than
    without any keyword search (zotero-rag-web, docs/spikes.md, S4).

    Ranking reads `PgStore.refresh_keyword_stats`'s statistics, and refreshes them
    first when a write has made them stale.
    """

    def __init__(self, store: PgStore) -> None:
        self._store = store

    def search(
        self, question: str, limit: int, filters: Filters | None = None
    ) -> list[StoredChunk]:
        if limit <= 0 or (filters is not None and filters.selects_nothing):
            return []
        tokens = tokenize(question)
        if not tokens:
            return []
        if self._store.keyword_stats_stale():
            self._store.refresh_keyword_stats()
        where, params = filters_sql(filters)
        conditions = "AND " + where.removeprefix("WHERE ") if where else ""
        params |= {
            "tokens": tokens,
            "query": " | ".join(sorted(set(tokens))),
            "limit": limit,
            "k1": BM25_K1,
            "b": BM25_B,
            "decimals": SCORE_DECIMALS,
        }
        select = _SELECT.replace("SELECT c.id,", "SELECT s.score, c.id,", 1).replace(
            "FROM chunks c", "FROM scored s\nJOIN chunks c ON c.id = s.id", 1
        )
        statement = (
            _BM25_SQL.format(filters=conditions)
            + select
            + "ORDER BY round(s.score::numeric, %(decimals)s) DESC, c.id"
        )
        with self._store._cursor("search the store") as cur:
            cur.execute(cast(Any, statement), params)
            return [_chunk(row) for row in cur.fetchall()]
