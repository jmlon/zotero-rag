-- The tables `zotero_rag_core.stores.pg.PgStore` reads and writes.
--
-- An application that owns the database (zotero-rag-web, through its migrations)
-- creates these itself and may add columns and tables of its own; `PgStore.create_schema`
-- runs this file for everyone else (the CLI's `--store postgres://...`, the tests).
-- Every statement is idempotent.

CREATE EXTENSION IF NOT EXISTS vector;

-- How the store was built (fingerprint.StoreFingerprint). Exactly one row.
CREATE TABLE IF NOT EXISTS chunk_store (
    id          smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    fingerprint jsonb    NOT NULL DEFAULT '{}'::jsonb
);
INSERT INTO chunk_store (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- One Zotero item: what a citation names and what filters select on. A NULL list
-- or year means "not recorded", which fails any filter on it; `year_int` 0 means
-- the item has no year (store.YEAR_UNKNOWN).
CREATE TABLE IF NOT EXISTS items (
    key             text PRIMARY KEY,
    item_type       text,
    title           text,
    authors         text,
    year            text,
    year_int        integer,
    doi             text,
    zotero_url      text,
    collection_keys text[],
    tags            text[]
);
CREATE INDEX IF NOT EXISTS items_collection_keys_idx ON items USING gin (collection_keys);
CREATE INDEX IF NOT EXISTS items_tags_idx ON items USING gin (tags);
CREATE INDEX IF NOT EXISTS items_year_int_idx ON items (year_int);

-- One indexed file or note of an item.
CREATE TABLE IF NOT EXISTS attachments (
    key             text   PRIMARY KEY,
    item_key        text   NOT NULL REFERENCES items (key) ON DELETE CASCADE,
    version         bigint,
    document_format text,
    collection_name text
);
CREATE INDEX IF NOT EXISTS attachments_item_key_idx ON attachments (item_key);

-- One embedded passage. `embedding` has no fixed width so that a model of another
-- width only needs `reset()`; the HNSW index casts to the width in use and is
-- created by PgStore once that width is known. `tsv` holds the passage's tokens as
-- `retrieval.tokenize` splits them, so keyword search matches what BM25 matches.
-- `extra` keeps any metadata the columns do not, so a chunk reads back as written.
CREATE TABLE IF NOT EXISTS chunks (
    id              text     PRIMARY KEY,
    attachment_key  text     NOT NULL REFERENCES attachments (key) ON DELETE CASCADE,
    page            integer,
    chunk_index     integer,
    locator_kind    text,
    locator_label   text,
    embedding_model text,
    text            text     NOT NULL,
    embedding       vector   NOT NULL,
    tsv             tsvector NOT NULL,
    extra           jsonb    NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS chunks_attachment_key_idx ON chunks (attachment_key);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

-- Keyword search ranks with BM25 (rank_bm25's BM25Okapi), computed in SQL. It needs
-- each chunk's length in tokens and, per term, how many chunks contain it. Those
-- statistics are rebuilt by PgStore.refresh_keyword_stats(); every write marks them
-- stale, and a search refreshes stale statistics before ranking.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS n_tokens integer;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS blank boolean;

CREATE TABLE IF NOT EXISTS term_df (
    word text    PRIMARY KEY,
    ndoc integer NOT NULL
);

ALTER TABLE chunk_store ADD COLUMN IF NOT EXISTS keyword_stale boolean NOT NULL DEFAULT true;
ALTER TABLE chunk_store ADD COLUMN IF NOT EXISTS keyword_docs integer NOT NULL DEFAULT 0;
ALTER TABLE chunk_store ADD COLUMN IF NOT EXISTS keyword_avgdl double precision NOT NULL DEFAULT 0;
ALTER TABLE chunk_store ADD COLUMN IF NOT EXISTS keyword_epsilon_idf double precision NOT NULL DEFAULT 0;
