# RAG CLI interface for Zotero

**Status:** PoC complete — `list-collections`, `init`, `index`, and `query` are
all built and verified against the live library; every acceptance criterion in
§12 passes. See **§14 Implementation status**.
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
- Closes with numbered next steps, beginning with the **`cd` into the folder**
  (omitted when the target already is the current directory), because `index` and
  `query` only run from inside an index folder. The `index` step shows **both**
  ways to name a collection — `--collection "NAME"` and `--collection-key KEY` —
  and notes that names match case-insensitively and that the key is what
  disambiguates a repeated name. When a `.env` exists further up, it also says
  that file will cover any variable left blank locally.
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
  root**, meaning every collection recursively. Items belonging to *no* collection
  are not included. This prints a warning and requires interactive confirmation;
  `--yes` skips the prompt for scripted runs.
- Subcollections are indexed **recursively** (`recursive: true` in config).
- `--dry-run` lists the items and attachments that would be indexed, with counts,
  and makes no embedding calls and no writes. It also **does not pin** the
  collection — pinning is a write, and a dry run performs none.
- Collection path matching is case-insensitive, and leading/trailing slashes are
  ignored, so `/DataManagement/Data Trust` and `datamanagement/data trust` both
  resolve.

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

CLI flag → index folder `./config.yaml` → packaged default `config.yaml`.

The environment carries **credentials only** (§5.2), never ordinary settings: a
value like `top_k` or `llm_model` comes from a CLI flag, the index folder's
`config.yaml`, or the packaged defaults it inherits from. Keeping the environment
out of that chain means there is exactly one file to look at when a setting is
not what you expected.

The default `config.yaml` ships as **package data** and is read via
`importlib.resources`, so `init` works when the tool is installed globally
(e.g. `uv tool install`) and not only from a source checkout.

### 5.2 Secrets

Secrets are **never** stored in `config.yaml` — only the *names* of the
environment variables that hold them, resolved at load time (following the
`ragbench.config.GlobalConfig.resolve_env()` pattern).

`.env` files are **layered**, nearest first: the current folder, then each parent
folder, then `~/.env`. A variable exported in the real environment wins over every
file, and among files the nearest **non-empty** value wins.

Entries with an empty value are ignored rather than treated as set. This is not a
detail: `init` writes a `.env` *template* whose values are blank, so under
python-dotenv's own first-match-wins search that template shadowed the real keys a
user kept in the parent folder, and `index` failed with "missing credentials"
while pointing at a file that looked correctly filled in. Layering plus the
empty-value rule is what makes the common setup — one `.env` beside several index
folders — work.

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
2b. Some files in Zotero storage are served with their **multipart upload
   envelope** still wrapped around the PDF bytes (`--boundary`, form-data
   headers, then `%PDF`), even though the API reports `Content-Type:
   application/pdf`. The envelope is stripped before anything is written to the
   cache, so the cache always holds valid PDFs. Files that already start with
   `%PDF` pass through untouched.
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
option. It also asks the model to answer **in the language of the question**,
which is what lets the Spanish questions of §12.1 be answered from English
passages.

### 8.2 Which sources are listed

Retrieval always returns `top_k` passages, but an "I don't know" answer cites
none of them. The printed source list therefore contains only the sources the
answer actually marked inline (`[1]`, `[2, 3]`); when it cites nothing, the CLI
says so and names the number of passages retrieved instead of listing them.
`--show-sources` overrides this and shows every retrieved passage, cited or not,
dimming the uncited ones. `--json` always carries all retrieved sources, each
with a `cited` boolean.

Chunks retrieved from the **same page of the same attachment** collapse into one
numbered source, so a page is never listed twice.

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
  structures; `ruff` and `mypy` (default strictness) must pass, run locally
  (`uv run ruff check . && uv run mypy`). Wiring them into CI is a future
  milestone (§11) — there is no CI pipeline in the PoC.
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
streaming output · metadata filters at query time (year, author, tag) · CI
running the test suite, `ruff`, and `mypy`.

## 12. Acceptance criteria

Test collection: **`Writing`**, key **`TGXVGXGP`**, 6 items of which **4 have PDF
attachments** (small, chosen for fast iteration).

1. `zotero-rag list-collections` prints the collection tree and shows `Writing`
   with key `TGXVGXGP` and its item count. ✅ *verified — 85 collections listed.*
2. `zotero-rag init .` in an empty directory creates `config.yaml`, a `.env`
   template, and a `.gitignore`, and contacts nothing.
   ✅ *verified — also refuses to re-init, and never overwrites an existing `.env`.*
3. Running `index` or `query` outside an index folder exits with a clear error.
   ✅ *verified for `index` — fails on the missing config.yaml before any network call.*
4. `zotero-rag index --collection-key TGXVGXGP --dry-run` lists the PDFs that
   would be indexed and makes no embedding API calls.
   ✅ *verified — 4 PDFs planned, 2 items warned as having no PDF; `--collection Writing`
   resolves identically.*
5. `zotero-rag index --collection-key TGXVGXGP` downloads the PDFs into `cache/`,
   populates `chroma/`, pins the collection into `config.yaml`, and prints a
   summary. Items without a PDF attachment produce warnings, not failures.
   ✅ *verified — 4 PDFs, 22 pages, 146 chunks, 30,863 embedding tokens in ~7s;
   the 2 item-less entries warn and the run still exits 0.*
6. A second `index` run indexes nothing new and reports all attachments skipped
   as already indexed.
   ✅ *verified — 0 indexed, 4 skipped, store still holds 146 chunks. Requesting a
   different collection in a pinned folder is refused.*
7. `zotero-rag query "<question>"` returns an answer grounded in the collection
   with **at least one correct Zotero citation** (title/author/year resolving to
   a real item in `Writing`), plus a working `zotero_url`.
   ✅ *verified — "¿Qué TIPs hay para escritores?" answered in Spanish from
   English passages, citing Driscoll & Aquilina (2011) p. 5, Elserag (2012) p. 1,
   and Powell (2010) p. 2, covering most of the expected points in §12.1.
   5 chunks, 1,287 prompt + 181 completion tokens, ~3s.*
8. A question with no support in the collection is answered with an explicit
   "I don't know" rather than a fabricated answer.
   ✅ *verified — a question about the boiling point of mercury answers "No sé."
   and lists no sources (§8.2).*

### 12.1 Sample questions

The `Writing` collection covers academic writing and the paper-publication
process. Questions are asked in **Spanish**, which also exercises the
multilingual concern in §5.3: if retrieval underperforms on these, the fix is to
override `embedding_model` with a multilingual OpenRouter slug.

A correct answer must cover most of the expected points **and** cite at least one
item that really exists in `Writing`.

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

The three questions and their expected points above are the source of truth; the
`Writing` collection is small enough that they are maintained here rather than in
a separate file.

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

## 14. Implementation status

Last updated: 2026-07-27. **182 tests passing; `ruff` and `mypy` clean.**

### 14.1 Done

| Milestone | Scope | State |
|---|---|---|
| 0 | Package skeleton, config layer, `list-collections` | ✅ verified live |
| 1 | `init` (offline scaffolding) | ✅ verified live |
| 2a | Collection resolution + `index --dry-run` | ✅ verified live |
| 2b | Download, extract, chunk, embed, store, pin | ✅ verified live |
| 3 | `query`: retrieve, prompt, answer, cite | ✅ verified live |

**All eight acceptance criteria (§12) are verified against the real library.**
Everything still open is a §11 future milestone, not PoC scope.

### 14.2 `query` as built (§3.4, §8)

- `llm.py` — `build_chat(config)` (`ChatOpenAI` against OpenRouter, `temperature`
  from config), plus `usage_from_message()` and `message_text()`, which flattens
  the block-style `content` some providers return.
- `querying.py` — `answer_question()`: embed the question, retrieve, number the
  sources, build the prompt, call the model. Both network seams (embedding, chat)
  are injected callables, as in `indexer.py`, so the pipeline is tested against a
  real on-disk Chroma store with no API access.
  - `collect_sources()` groups chunks into cited sources (§8.2) and tolerates
    chunks missing metadata fields.
  - `cited_numbers()` parses the inline `[n]` markers back out of the answer, so
    the CLI can list only what was really used.
  - An empty retrieval short-circuits to `NO_CONTEXT_ANSWER` with **no LLM call**.
  - Any exception from the chat client becomes an `LLMError`, printed as a
    one-line user-facing error.
- `cli.py` — `_open_index()` rejects a non-index folder (`NotAnIndexFolderError`)
  and one that was never indexed or holds zero chunks (`NotIndexedError`), both
  before any credential is needed. `--json` writes to stdout with the progress
  spinner diverted to stderr, so the payload is never contaminated.

### 14.3 Module map

```
src/zotero_rag/
  cli.py           typer app: list-collections, init, index, query
  config.py        pydantic models, layered loading, pin_collection()
  scaffold.py      init: config.yaml + .env + .gitignore
  zotero_api.py    client, collection tree, path/key resolution, item fetch
  planning.py      build_plan(): §7.1 selection rules, chunk_metadata()
  metadata.py      authors_str(), parse_year(), zotero_url()
  download.py      cache + multipart-envelope repair
  extraction.py    pypdf, page by page
  chunking.py      RecursiveCharacterTextSplitter per page
  embedding.py     OpenRouter embeddings + tiktoken counting
  vectorstore.py   chromadb wrapper, StoredChunk
  indexer.py       orchestration, IndexStats, failure isolation
  llm.py           OpenRouter ChatOpenAI + usage extraction
  querying.py      retrieval, citation numbering, prompt, QueryResult
  data/config.yaml packaged defaults (copied by init)
```

Tests mirror these; `tests/conftest.py` builds real PDFs with `pdf_factory`
(no reportlab — hand-assembled, honouring the no-heavy-deps rule).

### 14.4 Reproducing the verified state

```bash
uv sync --all-groups
uv run pytest -q && uv run ruff check . && uv run mypy

cp .env.example .env      # fill ZOTERO_LIBRARY_ID + ZOTERO_API_KEY (+ OPENROUTER_API_KEY)
uv run zotero-rag list-collections
mkdir /tmp/idx && cp .env /tmp/idx/ && cd /tmp/idx
uv run zotero-rag init .
uv run zotero-rag -v index --collection-key TGXVGXGP
uv run zotero-rag -v query "¿Qué TIPs hay para escritores?"
```

Expected: 4 PDFs, 22 pages, 146 chunks, ~30,900 embedding tokens, ~7s. A second
`index` reports 0 indexed / 4 skipped. The query answers in Spanish in ~3s,
citing 2–3 real `Writing` items with working `zotero.org` URLs.

**Second collection, verified from a global `uv tool install`:** `Networking`
(`SHWKWYDD`), which is a *tree* rather than a single collection — 8 collections, 18
items — indexed 13 PDFs into 1,150 chunks from 182 pages, skipping 10 attachments
of unsupported content type, 5 items with no PDF, and 1 linked URL. A query about
fog networks answered from two pages of the same survey, each its own numbered
source per §8.2. This is the run that exercises recursion, the §7.1 skip
categories, and a store an order of magnitude larger than `Writing`.

### 14.5 Findings and open items

- **Corrupt files in Zotero storage.** All 4 PDFs in `Writing` are served with
  their multipart upload envelope wrapped around the PDF bytes, despite a
  `Content-Type: application/pdf` header. Confirmed with a raw `requests` call,
  so it is neither our bug nor pyzotero's — something uploaded them that way and
  it may affect other collections. `download.py` strips the envelope (verified
  lossless: identical page and character counts). Worth checking whether Zotero's
  own reader opens them.
- **API drift from the reference code:** pyzotero renamed `ResourceNotFound` to
  `ResourceNotFoundError`; the code uses `zot.file()` rather than `zot.dump()` so
  bytes can be sanitised before reaching disk.
- **Decisions recorded rather than asked:** `--dry-run` does not pin (pinning is
  a write); whole-library root means every collection recursively and excludes
  items in no collection; `init` also writes a `.gitignore` (§3.2); the source
  list shows only cited sources (§8.2); chunks sharing a page collapse into one
  source; the prompt asks for an answer in the question's language.
- **Corrections made to this PRD after the audit that closed milestone 3:** §5.1
  claimed an environment-variable layer for ordinary settings that was never
  built and is not wanted — the environment now carries credentials only. §10's
  "in CI" became "run locally", with CI moved to §11. §12.1's reference to a
  `sample-questions.txt` that never existed was dropped in favour of the
  questions listed there. Two real drifts were fixed in code instead: `-v` is now
  accepted on `index` and `query` themselves (it previously had to precede the
  subcommand, contradicting §3.3/§3.4), and the index summary now reports items
  seen, which §10 requires.
- **`VectorStore.query()` and `fetch_username()` are now exercised live:** the
  citation URLs the `query` command printed
  (`zotero.org/jmlondono/collections/TGXVGXGP/items/…/collection`) resolve to the
  real items.
- **Retrieval quality with the default embedding model is adequate for Spanish
  questions over English text** — no multilingual override was needed (§5.3). The
  answers do lean on whichever 5 chunks come back, so a question whose expected
  points are spread across four papers is answered from two or three of them; a
  reranker or a larger `top_k` is the §11 lever if that matters.
