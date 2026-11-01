# zotero-rag-core

The library behind [`zotero-rag`](../../README.md): reading a Zotero library,
planning and extracting its attachments, chunking, embedding, hybrid retrieval,
reranking, and answering questions with citations.

It has no command line and no terminal UI. Applications build a
`zotero_rag_core.config.Config` however suits them and call the pipeline:
the `zotero-rag` CLI loads YAML index folders, and `zotero-rag-web` reads
environment variables.

Extras:

- `chroma` — the persistent Chroma store an index folder keeps (`zotero_rag_core.stores.chroma.ChromaStore`), one implementation of `zotero_rag_core.store.ChunkStore`.
- `pgvector` — a shared PostgreSQL store with the pgvector extension
  (`zotero_rag_core.stores.pg.PgStore`), and keyword search in the database
  (`PgFullTextSearcher`), which ranks with BM25 computed in SQL and returns what
  the in-memory `BM25Searcher` returns. The tables it expects are in
  `stores/pg_schema.sql`; `PgStore.from_url(url, create=True)` creates them.
  Embedding models must have at most 2,000 dimensions, pgvector's index limit.

## Tests

The PostgreSQL tests start a disposable `pgvector/pgvector:pg17` container through
Docker, or use the database in `ZOTERO_RAG_TEST_PG_URL` (which they empty). Without
either they are skipped; set `ZOTERO_RAG_REQUIRE_PG=1` to make that a failure, as CI
should.
