# RAG CLI interface for Zotero

**Status:** Draft — Proof of Concept (PoC) scope
**Version:** 0.1.0

## 1. Goal

A command line utility for indexing and querying a Zotero bibliographic database
using RAG techniques. The user selects one Zotero collection, indexes the PDF
attachments of its items into a local vector database, and then asks natural
language questions that are answered by an LLM grounded in the retrieved
passages, with citations back to the Zotero items.

### 1.1 Non-goals for the PoC

The following are explicitly **out of scope** for this version and recorded as
future milestones (§11):

- Group libraries (user library only).
- Vector stores other than Chroma (Pinecone is a planned milestone).
- Rerankers, hybrid / BM25 retrieval, windowed retrieval, agentic or multi-step RAG.
- OCR of scanned PDFs.
- Ingestion of HTML snapshots, EPUBs, linked URLs, and Zotero notes.
- `--force` / clean-slate reindex.
- Detecting items updated or removed in Zotero after indexing.
- Custom system prompt from file.
- Cost estimation and pricing tables.
- Batched / concurrent embedding calls.
- Interactive REPL, streaming output, web UI, MCP server.

## 2. Definitions

- **Index folder** — a directory containing a `config.yaml`. Its presence is what
  marks the folder as an index. `index` and `query` must be run from inside an
  index folder; the CLI reports an error otherwise.
- **Collection path** — a path through the Zotero collection *hierarchy*, e.g.
  `Research/Data Trust/Legal`. Not a filesystem path.
- **Pinned collection** — the collection recorded in the index `config.yaml` by
  the first successful `index` run. An index folder serves exactly one collection.

## 3. CLI surface

Console script: `zotero-rag`. Built with `typer`; progress and tables via `rich`.

### 3.1 `list-collections`

```
zotero-rag list-collections [--depth N]
```

- Prints the Zotero collection hierarchy as a tree, showing each collection's
  **name**, **key**, and **item count**, so the user can copy either the path or
  the key for use with `index`.
- Runs **outside** an index folder (it is normally used before `init`), so it
  resolves credentials from `.env` and the packaged default `config.yaml` only.

### 3.2 `init`

```
zotero-rag init [PATH]
```

- `PATH` defaults to `.`; `zotero-rag init myindex` creates that directory.
- Writes `config.yaml` into the folder, copied from the packaged default config
  (§5) so every setting is present and editable.
- Writes a `.env` template with the required variable names and empty values for
  the user to fill in. The names come from the config, so a customised `*_env`
  setting yields a matching template.
- Writes a `.gitignore` covering `.env`, `chroma/`, and `cache/` — an index
  folder holds a secrets file and is often created inside a repository.
- Does **not** contact Zotero, does **not** pin a collection, and does **not**
  create the vector database.
- Refuses to run if `config.yaml` already exists. An existing `.env` or
  `.gitignore` is left untouched and reported as kept, never overwritten.

### 3.3 `index`

```
zotero-rag index [--collection PATH | --collection-key KEY] [--dry-run] [--yes] [-v]
```

- **First run** pins the target collection into `config.yaml`
  (`collection_key`, `collection_name`, `collection_path`) and creates the vector
  store and cache directories.
- **Subsequent runs** take no collection argument and reuse the pinned
  collection. Passing a `--collection`/`--collection-key` that disagrees with the
  pinned one is an error (changing it will require the future `--force`).
- If neither option is given on the first run, the target is the **whole library
  root**. This prints a warning and requires interactive confirmation; `--yes`
  skips the prompt for scripted runs.
- Subcollections are indexed **recursively** (`recursive: true` in config).
- `--dry-run` lists the items and attachments that would be indexed, with counts,
  and makes no embedding calls and no writes.

### 3.4 `query`

```
zotero-rag query "QUESTION" [--top-k N] [--show-sources] [--json] [-v]
```

- One-shot query. Retrieves `top_k` chunks, calls the LLM, prints the answer
  followed by a numbered source list.
- `--show-sources` additionally prints the full text of each retrieved chunk.
- `--json` emits a machine-readable object (`answer`, `sources[]`, `usage`).
- `--top-k` overrides the configured value for this call.

## 4. Zotero access

- Online Zotero Web API via `pyzotero`, authenticated with an API token.
- `library_type` is `user` for the PoC.
- Collection path resolution: split the path on `/` and walk the hierarchy from
  the top-level collections. If any segment matches more than one collection at
  its level, report an error listing the candidate keys and exit; the user then
  disambiguates with `--collection-key`.

## 5. Configuration

### 5.1 Precedence

CLI flag → environment variable → index folder `./config.yaml` → packaged default
`config.yaml`.

The default `config.yaml` ships as **package data** and is read via
`importlib.resources`, so `init` works when the tool is installed globally
(e.g. `uv tool install`) and not only from a source checkout.

### 5.2 Secrets

Secrets are **never** stored in `config.yaml` — only the *names* of the
environment variables that hold them, resolved at load time (following the
`ragbench.config.GlobalConfig.resolve_env()` pattern). `.env` is discovered by
`python-dotenv`'s normal search (current folder, then upward / home).

Required variables: `OPENROUTER_ENDPOINT`, `OPENROUTER_API_KEY`,
`ZOTERO_LIBRARY_ID`, `ZOTERO_API_KEY`.

### 5.3 Schema

Modelled with pydantic `BaseModel`s, mirroring `ragbench/config.py`.

```yaml
global:
  # Only env-var NAMES live here; values are resolved from the environment.
  endpoint_env: OPENROUTER_ENDPOINT
  api_key_env: OPENROUTER_API_KEY
  zotero_library_id_env: ZOTERO_LIBRARY_ID
  zotero_api_key_env: ZOTERO_API_KEY
  zotero_library_type: user
  defaults:
    llm_model: openai/gpt-4o-mini
    embedding_model: openai/text-embedding-3-small
    vector_db: chroma
    reranker: null

index:
  # Pinned by the first `index` run; null in a freshly initialized folder.
  collection_path: null
  collection_key: null
  collection_name: null
  recursive: true

  chunk_size: 1000        # characters (RecursiveCharacterTextSplitter)
  chunk_overlap: 200      # characters
  top_k: 5
  temperature: 0.0

  # null => inherit from global.defaults
  llm_model: null
  embedding_model: null
  vector_db: null
```

Multilingual corpora are supported by overriding `embedding_model` with a
multilingual OpenRouter slug. No local embedding models (see §9).

## 6. Index folder layout

```
<index-folder>/
  config.yaml        # marker + settings + pinned collection
  .env               # created by init as a template (never committed)
  .gitignore         # created by init: ignores .env, chroma/, cache/
  chroma/            # Chroma persistent store (created on first index)
  cache/             # downloaded attachments, <attachment-key>.pdf
```

The cache is retained so re-indexing does not re-download files. The Chroma
collection name is fixed (`zotero_rag`).

## 7. Ingestion pipeline

For the pinned collection, recursively:

1. Fetch collection items via `pyzotero`.
2. Separate parent items from `attachment` items; cache parents by key so each
   attachment can resolve its bibliographic metadata.
3. For each attachment with `contentType == application/pdf`: if the file is not
   already in `cache/`, download it with `zot.dump()`.
4. Extract text **page by page** with `pypdf`, then split each page's text with
   `RecursiveCharacterTextSplitter`. Splitting per page keeps the recorded page
   number exact.
5. Embed chunks (`langchain-openai` `OpenAIEmbeddings` against the OpenRouter
   endpoint, `check_embedding_ctx_length=False` — OpenRouter slugs are not in
   `tiktoken`'s registry) and upsert into Chroma.

### 7.1 Selection rules

- **PDF attachments only.** `linked_url`, HTML snapshots, EPUBs, and notes are
  skipped and counted in the run summary.
- Items with **no PDF attachment** are not indexed; a warning names the item.
- Items with **multiple PDF attachments**: all of them are indexed.
- **Scanned / image-only PDFs** (extraction yields no text) are skipped with a
  warning; no OCR.
- An item reachable through two subcollections is embedded **once** —
  deduplicated by `attachment_key`.

### 7.2 Chunk metadata

Stored per chunk, used for both citation and future filtering:

| field | source |
|---|---|
| `item_key` | parent item key |
| `attachment_key` | attachment key |
| `title` | parent item `title` |
| `authors` | `creators`, formatted `Last, First; Last, First` |
| `year` | parsed from the item's `date` field (multiple formats) |
| `doi` | item `DOI`, `""` if absent |
| `zotero_url` | `https://www.zotero.org/<user>/collections/<col>/items/<key>/collection` |
| `collection_name` | the (sub)collection the item was found in |
| `page` | 1-based PDF page number |
| `chunk_index` | index of the chunk within that page |

Chroma document id: `<attachment_key>:<page>:<chunk_index>`.

### 7.3 Incremental indexing

- Before processing an attachment, query Chroma for existing chunks with that
  `attachment_key`. If any exist, the attachment is **skipped**.
- The attachment key alone is the criterion — Zotero item `version` is not
  tracked in the PoC. Consequence, accepted for now: a PDF replaced or edited in
  Zotero after indexing is **not** re-indexed. Re-indexing changed items requires
  the future `--force`.
- Removal of items from Zotero is not propagated (future milestone).

## 8. Query pipeline

1. Load the index config, open the persistent Chroma collection (error if the
   folder has never been indexed).
2. Embed the question and retrieve `top_k` chunks by similarity.
3. Build the context and call the LLM (`langchain-openai` `ChatOpenAI` against
   OpenRouter, `temperature` from config).
4. Print the answer, then the numbered source list: title, authors, year, page,
   Zotero URL.

### 8.1 Prompt requirements

The system prompt instructs the model to:

- Answer using **only** the provided context.
- Say it does not know when the context is insufficient — never speculate.
- **Always cite** the sources it used, with inline markers (`[1]`, `[2]`) keyed to
  the numbered source list.
- Be concise.

The prompt is fixed in the PoC; loading a custom prompt from a file is a future
option.

## 9. Dependencies

Hard constraint: **no heavyweight dependencies** — no HuggingFace libraries, no
PyTorch, no sentence-transformers. Embeddings are always remote via OpenRouter.

Runtime: `pyzotero`, `chromadb`, `pypdf`, `langchain-core`, `langchain-openai`,
`langchain-text-splitters`, `typer`, `rich`, `pydantic`, `pyyaml`,
`python-dotenv`, `tiktoken` (token counts for `-v`).

`chromadb` is used **directly** (no `langchain-chroma` wrapper). The `langchain`
meta-package is not installed.

`pyproject.toml` changes: remove `chroma` (an unrelated color library — the
vector DB is `chromadb`); keep `pinecone` as a planned future milestone.

## 10. Non-functional requirements

- **Scale (PoC):** tens to ~100 PDFs per collection. Serial embedding calls are
  acceptable; batching and concurrency are a future milestone.
- **Failure policy:** any single item, attachment, download, or extraction
  failure logs a `WARNING` and processing continues. The run ends with a summary:
  items seen, attachments indexed, chunks written, skipped (by reason), failed.
- **Logging:** standard `logging`; `-v/--verbose` reports elapsed time and token
  usage (embedding tokens for `index`, prompt/completion tokens for `query`).
  Cost estimation is out of scope.
- **Type safety:** pydantic models for config and for the item/chunk/result
  structures; `ruff` and `mypy` (default strictness) in CI.
- **Python:** ≥ 3.12, `src` layout, package `zotero_rag`, entry point
  `zotero-rag = "zotero_rag.cli:app"`.
- **Tests:** no `pyzotero` mocks. Unit tests cover config loading and precedence,
  collection-path resolution, chunking, metadata construction, and citation
  formatting. The end-to-end path is validated by the manual smoke test in §12
  once the environment variables are configured.

## 11. Future milestones

Group libraries · Pinecone and other vector stores · rerankers · hybrid /
BM25 retrieval · windowed and hierarchical retrieval · agentic RAG · `--force`
clean-slate reindex · version-aware incremental indexing · deletion sync · OCR ·
HTML / EPUB / notes ingestion · custom system prompt from file · cost
estimation · batched and concurrent embedding · interactive REPL with history ·
streaming output · metadata filters at query time (year, author, tag).

## 12. Acceptance criteria

Test collection: **`Writting`**, key **`TGXVGXGP`**, 6 items (small, chosen for
fast iteration).

1. `zotero-rag list-collections` prints the collection tree and shows `Writting`
   with key `TGXVGXGP` and its item count. ✅ *verified — 85 collections listed.*
2. `zotero-rag init .` in an empty directory creates `config.yaml`, a `.env`
   template, and a `.gitignore`, and contacts nothing.
   ✅ *verified — also refuses to re-init, and never overwrites an existing `.env`.*
3. Running `index` or `query` outside an index folder exits with a clear error.
4. `zotero-rag index --collection-key TGXVGXGP --dry-run` lists the PDFs that
   would be indexed and makes no embedding API calls.
5. `zotero-rag index --collection-key TGXVGXGP` downloads the PDFs into `cache/`,
   populates `chroma/`, pins the collection into `config.yaml`, and prints a
   summary. Items without a PDF attachment produce warnings, not failures.
6. A second `index` run indexes nothing new and reports all attachments skipped
   as already indexed.
7. `zotero-rag query "<question>"` returns an answer grounded in the collection
   with **at least one correct Zotero citation** (title/author/year resolving to
   a real item in `Writting`), plus a working `zotero_url`.
8. A question with no support in the collection is answered with an explicit
   "I don't know" rather than a fabricated answer.

### 12.1 Sample questions

The `Writting` collection covers academic writing and the paper-publication
process. Questions are asked in **Spanish**, which also exercises the
multilingual concern in §5.3: if retrieval underperforms on these, the fix is to
override `embedding_model` with a multilingual OpenRouter slug.

A correct answer must cover most of the expected points **and** cite at least one
item that really exists in `Writting`.

1. **¿Qué TIPs hay para escritores?**
   Expected points: uno es tan bueno como su último paper (el éxito previo no
   garantiza el futuro); hay que enganchar al editor con el abstract; no borrar
   archivos, guardar versiones previas; la curva de aprendizaje es larga; el
   trabajo más significativo es eliminar excesos y simplificar; escribir todos
   los días; terminar cuando la idea ya se transmitió.

2. **¿Cuáles son las tareas previas a la escritura de un paper?**
   Expected points: revisar bibliografía y referencias, levantar el estado del
   arte; tener los resultados de la investigación y su análisis estadístico;
   preparar figuras y representaciones; identificar la revista objetivo.

3. **¿Qué pasos involucra el proceso de publicación?**
   Expected points: revisiones solicitadas por los pares; carta de aceptación
   del editor; revisión del *galley proof* (contenido, tablas, organización,
   tipografía).

Source of truth for the expected points: `sample-questions.txt`.

## 13. Reference material

- [pyzotero](https://pyzotero.readthedocs.io/en/latest/)
- [Pinecone Python SDK](https://docs.pinecone.io/reference/sdks/python/overview) (future milestone)

Example code that may be reused:

- @/home/jmlon/GIT/AWS/UPB/Python/langchain/zotero/webapi.ipynb — Zotero Web API usage
- @/home/jmlon/GIT/AWS/UPB/Python/langchain/zotero/zotero_ingest.py — collection walk, attachment handling, metadata extraction, `authors_str`, `parse_year`
- @/home/jmlon/GIT/AWS/RagEvaluator/ragbench/systems/simple_rag.py — index/query structure, additive indexing, usage accounting
- @/home/jmlon/GIT/AWS/RagEvaluator/ragbench/llm.py — OpenRouter-backed `ChatOpenAI` / `OpenAIEmbeddings` factories
- @/home/jmlon/GIT/AWS/RagEvaluator/ragbench/config.py — pydantic config models, env-var indirection for secrets
- @/home/jmlon/GIT/AWS/UPB/Python/langchain/zotero/rag_agent.py — prompt and chain shape (note: uses a reranker, out of PoC scope)
