# RAG CLI interface for Zotero

**Status:** PoC complete — `list-collections`, `init`, `index`, `query` and `eval`
are all built and verified against the live library; every acceptance criterion in
§12 passes. See **§14 Implementation status** for what was built and
**§11 Roadmap** for the sequenced milestones that follow.
**Version:** 0.4.0

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
- OCR of scanned PDFs. *(Still open; see the deferral note in §11.2.)*
- Ingestion of HTML snapshots, EPUBs, linked URLs, and Zotero notes.
  *(Snapshots, EPUBs and notes delivered in M5, §11.2 — plus text/Markdown/CSV,
  DOCX and ODT. Linked URLs remain deferred.)*
- `--force` / clean-slate reindex. *(Delivered in M4, §11.1.)*
- Detecting items updated or removed in Zotero after indexing.
  *(Delivered in M4, §11.1.)*
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
zotero-rag index [--collection PATH | --collection-key KEY] [--dry-run] [--yes]
                 [--force [--refetch]] [--repair] [--prune] [-v]
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
- **Re-indexing (M4, §11.1).** A normal run re-embeds an attachment whose Zotero
  `version` is newer than the stored one, dropping its old chunks first.
  `--force` discards the store and re-embeds from `cache/`; adding `--refetch`
  also clears the cache so files are downloaded again. `--repair` re-embeds only
  the chunks a superseded model wrote, reusing their stored text — no downloads and
  no re-chunking. `--prune` deletes chunks whose attachment has left the
  collection; without it those are reported but kept.

### 3.4 `query`

```
zotero-rag query "QUESTION" [--top-k N] [--window N] [--retrieval MODE]
                            [--reranker none|cohere] [--show-sources] [--json] [-v]
```

- One-shot query. Retrieves `top_k` chunks, calls the LLM, prints the answer
  followed by a numbered source list.
- `--show-sources` additionally prints the full text of each retrieved chunk.
- `--json` emits a machine-readable object (`answer`, `sources[]`, `usage`,
  `retrieval`). The last carries `chunks_retrieved` and `chunks_in_context`, which
  differ once `--window` is in use.
- `--top-k` overrides the configured value for this call.
- `--window N` also sends the N chunks either side of each hit, in reading order
  (§11.4.1). Defaults to `retrieval_window`, which is `0` — measurement did not
  support making it the default.
- `--retrieval vector|hybrid` picks the strategy (§11.4.2). `hybrid` fuses a BM25
  keyword ranking with the vector one and **is the default**, because it measured
  better at no token cost. It is a query-time choice: switching it never requires
  re-indexing.
- `--reranker cohere` re-scores the retrieved candidates against the question and
  keeps the best `top_k` (§11.4.3), on top of whichever `--retrieval` mode ran.
  `--reranker none` disables a configured one. Off by default because it needs
  `COHERE_API_KEY`; recommended where that key exists, since `top_k: 5` with
  reranking beat `top_k: 10` without it on half the context.

### 3.4.1 `zotero-rag-ui`

```
zotero-rag-ui [--top-k N] [--window N] [--retrieval MODE] [--reranker none|cohere] [-v]
```

A separate console script with a full-screen terminal UI over the same pipeline as
`query`. It adds a folder tree for limiting a question to a sub-folder, streamed
Markdown answers and a persistent prompt history. Its flags mean what they mean for
`query`, and outside an index folder it fails with the same error and exit code.
Specified in `PRD-CursesUI.md`.

### 3.5 `eval`

```
zotero-rag eval [--questions FILE] [--top-k N] [--window N] [--retrieval MODE]
                [--reranker none|cohere] [--only ID]... [--assess-context]
                [--json] [-v]
```

- Runs a golden set of questions (§11.3) against the index and scores it: a table
  of one row per question, then a single summary line. Requires an index folder
  that has been indexed, like `query`.
- Reports, per question, the **grade** (a 5-level ordinal), **point coverage**,
  whether an expected item **reached `top_k`** (hit) and whether the answer
  actually **cited** it, plus tokens and elapsed time.
- The run prints the settings behind it — models, `top_k`, chunk size,
  temperature, retrieval mode, store size — so two runs can be compared honestly.
- `--questions` overrides the configured `eval.questions_file`; `--only` runs a
  single question by id and is repeatable; `--assess-context` adds one judge call
  per question rating whether the retrieved passages could support the expected
  answer at all, which separates a retrieval failure from a generation one.
- `--json` emits the whole report (`settings`, `aggregate`, `results[]`) on
  stdout, with progress diverted to stderr.

### 3.6 `generate-questions`

```
zotero-rag generate-questions [--out PATH] [--per-document N] [--language LANG]
                              [--only-item KEY]... [--limit N] [--force]
                              [--dry-run] [--yes] [-v]
```

- Writes a starting `questions.yaml` (§11.3.2) from the documents **already
  indexed**, so writing a golden set by hand is no longer the price of measuring a
  collection. One LLM call per document.
- Questions come from the **vector store, not `cache/`**: a question written from
  unindexed text could never be answered from the index, and would sit in the set
  as a permanent false failure. Each question therefore carries the exact
  `expected_items` key of the item its text came from.
- `--per-document` (default `eval.questions_per_document`, 3) sets how many;
  `--language` defaults to the language of each document, so `--language Spanish`
  is how §12.1's Spanish-over-English setup is reproduced.
- `--dry-run` lists the documents with chunk and character counts and an estimated
  prompt-token cost, and writes nothing. Above 30 documents an ordinary run asks
  for confirmation, as a whole-library `index` does; `--yes` skips it.
- **Refuses to overwrite an existing question set** unless `--force` is given, and
  refuses *before* making any model call. `--out` writes elsewhere.
- `--only-item` (repeatable) and `--limit` narrow the run, which is how a trial on
  one paper is done before spending on a whole collection.
- The result is a **draft to edit**, not a finished golden set, and the file's own
  header says so — see the end of §11.3.2 for why.

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

**Conditionally required:** `COHERE_API_KEY`, needed only when `reranker: cohere` is
enabled (§11.4.3). It is asserted when the reranker is built, so a folder that does
not use reranking never demands it — and one that does fails immediately rather than
part-way through a run. Cohere is a second provider because OpenRouter proxies chat
completions and embeddings but not reranking; every other credential still routes
through OpenRouter as §5.1 intends.

### 5.3 Schema

Modelled with pydantic `BaseModel`s, mirroring `ragbench/config.py`.

```yaml
global:
  # Only env-var NAMES live here; values are resolved from the environment.
  endpoint_env: OPENROUTER_ENDPOINT
  api_key_env: OPENROUTER_API_KEY
  zotero_library_id_env: ZOTERO_LIBRARY_ID
  zotero_api_key_env: ZOTERO_API_KEY
  cohere_api_key_env: COHERE_API_KEY   # only needed if a reranker is enabled
  zotero_library_type: user
  defaults:
    llm_model: openai/gpt-4o-mini
    embedding_model: openai/text-embedding-3-small
    vector_db: chroma
    reranker: null            # null | cohere; inherited by the index (§11.4.3)

index:
  # Pinned by the first `index` run; null in a freshly initialized folder.
  collection_path: null
  collection_key: null
  collection_name: null
  recursive: true

  chunk_size: 1000        # characters (RecursiveCharacterTextSplitter)
  chunk_overlap: 200      # characters
  top_k: 10               # 5 before 0.4.4; raised on M7's evidence (§11.4.1)
  retrieval_window: 0     # neighbours either side of each hit (M7, §11.4.1)
  retrieval: hybrid       # vector | hybrid (BM25 fused with vectors, §11.4.2)
  reranker: null          # null => inherit global.defaults.reranker (§11.4.3)
  reranker_model: rerank-v4.0-fast
  reranker_fanout: 4      # candidates retrieved per final passage
  temperature: 0.0

  # Which attachment wins when an item has several readable ones (§7.1).
  attachment_preference: [pdf, docx, odt, epub, html, txt]
  index_all_attachments: false
  index_notes: true

  # null => inherit from global.defaults
  llm_model: null
  embedding_model: null
  vector_db: null

eval:                       # M6, §11.3
  questions_file: questions.yaml   # relative paths resolve inside the index folder
  judge_model: null         # null => grade with the answering model
  judge_temperature: 0.0
  context_char_budget: 12000       # cap on the context handed to --assess-context

  # `generate-questions` (§11.3.2)
  generator_model: null     # null => write questions with the answering model
  questions_per_document: 3
  document_char_budget: 30000      # per document; longer ones are sampled head-and-tail
```

Multilingual corpora are supported by overriding `embedding_model` with a
multilingual OpenRouter slug. No local embedding models (see §9).

## 6. Index folder layout

```
<index-folder>/
  config.yaml        # marker + settings + pinned collection
  .env               # created by init as a template (never committed)
  .gitignore         # created by init: ignores .env, chroma/, cache/, .zotero-rag-ui/
  questions.yaml     # optional golden set for `eval` (§11.3); worth committing
  chroma/            # Chroma persistent store (created on first index)
  cache/             # downloaded attachments, <attachment-key>.<pdf|html|docx|…>
  collections.json   # the indexed collection tree, written by every `index` run
                     # once membership is current (PRD-CursesUI §4.1)
  .zotero-rag-ui/    # zotero-rag-ui state: `history` (one JSON string per line,
                     # last 1000 prompts) and `ui.log` under -v (PRD-CursesUI §7, §8)
```

The cache is retained so re-indexing does not re-download files. PDFs keep the
`.pdf` suffix so caches written before M5 still hit. The Chroma
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
3. Decide which attachment of each item to index (§7.1), then, if its file is not
   already in `cache/<key><suffix>`, download it. Notes need no download: their
   body travels in the item payload.
4. Extract text **segment by segment** — pages for a PDF via `pypdf`, sections for
   the other formats — then split each segment's text with
   `RecursiveCharacterTextSplitter`. Splitting per segment keeps the recorded
   locator exact (§7.1.1).
5. Embed chunks (`langchain-openai` `OpenAIEmbeddings` against the OpenRouter
   endpoint, `check_embedding_ctx_length=False` — OpenRouter slugs are not in
   `tiktoken`'s registry) and upsert into Chroma.

### 7.1 Selection rules

*Extended by M5 (§11.2), which added every format below except PDF.*

- **Readable formats:** PDF, HTML snapshots, plain text / Markdown / CSV, DOCX,
  ODT, EPUB, and Zotero notes. `linked_url` attachments and anything else
  (images, video, archives) are skipped and counted in the run summary **with
  their content type**, so the count is actionable.
- **Only files Zotero can serve are candidates.** Zotero sends `md5: null` for a
  file that was never uploaded (file sync off, storage quota full, WebDAV storage,
  or a sync still in progress), and the Web API answers 404 for it. Such attachments
  are skipped as `file not uploaded to Zotero storage`, and `linked_file`
  attachments, which live only on the user's disk, as `linked file`. Both are listed
  by paper title in the summary, with how to fix them in Zotero. A payload with no
  `md5` key is still tried. The check runs **before** the format preference, so an
  unsynced PDF gives way to a synced snapshot instead of winning and failing to
  download. An item left with nothing downloadable is not also reported as "no
  readable attachment".
- **One format per item.** When an item offers several readable attachments, the
  first format in `attachment_preference` wins and the rest are recorded as
  skipped. This exists because ~100 items in the target library hold both a PDF
  and an HTML snapshot of the same article; indexing both would embed that
  content twice and let the copies compete in retrieval.
  `index_all_attachments: true` indexes every readable attachment instead.
- Items with **several attachments of the winning format** — a paper and its
  supplement, say — have **all of them** indexed. The choice is between formats,
  not between files.
- **Notes are indexed alongside** an item's file, never instead of it, and take no
  part in the preference: a note is the reader's own commentary, so it complements
  the paper. `index_notes: false` turns them off. A *standalone* note (no parent
  item) is skipped, since it has no bibliographic metadata to cite.
- Items with **no readable attachment** are not indexed; a warning names the item.
- **Scanned / image-only PDFs** (extraction yields no text) are skipped with a
  warning; no OCR yet (§11.2).
- An item reachable through two subcollections is embedded **once** —
  deduplicated by `attachment_key`.

### 7.1.1 Locators

Only a PDF has pages, so the page number generalises to a **locator**: an ordinal,
a `kind`, and an optional label.

| Format | Locator kind | Ordinal counts | Label |
|---|---|---|---|
| PDF | `page` | pages | — |
| HTML, DOCX, ODT, EPUB, Markdown | `section` | heading-delimited sections | the heading (or the `<title>` for a page with none) |
| Zotero note | `note` | the note | — |
| plain text without headings | `section` | the whole file | — |

Citations render each kind in its own terms — `p. 4`, `§ Architecture`, `note` —
rather than pretending a snapshot has pages. Chunks written before M5 carry no
kind and are read as pages, which is what they are.

HTML extraction drops non-content elements outright (`script`, `style`, `nav`,
`footer`, `aside`, forms) because a snapshot is a whole web page and most of its
markup is furniture.

### 7.2 Chunk metadata

Stored per chunk, used for citation and for the terminal UI's folder filter:

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
| `collection_keys` | every indexed collection holding the parent item, as a list of strings. Refreshed on every run, with no re-embedding, because a move between folders does not bump the attachment version. Filter with `$contains`, never `$in` (PRD-CursesUI §4.3.1) |
| `page` | the locator ordinal: a 1-based page for a PDF, a section number otherwise. The key keeps its name so chunks written before M5 are not stranded |
| `chunk_index` | index of the chunk within that segment |
| `locator_kind` | `page` \| `section` \| `paragraph` \| `note` \| `chunk` (M5) |
| `locator_label` | the section heading, when the format has one (M5) |
| `document_format` | `pdf` \| `html` \| `txt` \| `docx` \| `odt` \| `epub` \| `note` (M5) |
| `embedding_model` | the model that produced this vector (M4) |
| `attachment_version` | the attachment's Zotero version when embedded (M4) |

Chroma document id: `<attachment_key>:<page>:<chunk_index>`.

### 7.3 Incremental indexing

*Superseded by M4 (§11.1); this is what a run now does.*

- Before processing an attachment, query Chroma for existing chunks with that
  `attachment_key`. If none exist, it is indexed.
- If chunks exist, the attachment's Zotero `version` is compared with the
  `attachment_version` recorded on those chunks. A newer version means the file or
  its metadata changed, so the old chunks are **deleted** and the attachment is
  re-embedded; otherwise it is **skipped**. Deleting first matters: a shorter
  replacement would otherwise leave the tail pages of the old file behind.
- A stored version of `0` (chunks written before versions were recorded) counts as
  unchanged, so upgrading does not silently re-embed every existing store.
- Chunks whose attachment is no longer in the pinned collection are reported, and
  removed by `--prune`. One set difference covers three cases: deleted in Zotero,
  moved out of the collection, and no longer a supported content type.

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
numbered source, so a page is never listed twice. Two pages of the *same* document
remain two sources, because a citation names a page.

Source numbers therefore stay tied to the passages as they were given to the
model, which means the printed list can have **gaps** — an answer citing `[2]` and
`[4]` lists only 2 and 4. This is deliberate: renumbering to 1..n would mean
rewriting the markers inside the model's own text, which can only drift out of
sync with what it wrote.

## 9. Dependencies

Hard constraint: **no heavyweight dependencies** — no HuggingFace libraries, no
PyTorch, no sentence-transformers. Embeddings are always remote via OpenRouter, and
reranking is remote via Cohere for the same reason: a local cross-encoder is the one
obvious way to implement §11.4.3 and it would breach this rule outright.

Runtime: `pyzotero`, `chromadb`, `pypdf`, `langchain-core`, `langchain-openai`,
`langchain-text-splitters`, `typer`, `rich`, `pydantic`, `pyyaml`,
`python-dotenv`, `tiktoken` (token counts for `-v`), `httpx` (imported directly to
catch the transport errors `pyzotero` lets through — see §14.7), `rank-bm25`
(keyword retrieval, §11.4.2 — pure Python over numpy, which `chromadb` already
requires, so it adds no new native dependency). It ships no type stubs, so `mypy`
has one scoped `ignore_missing_imports` override for it and no others.

HTML, DOCX, ODT and EPUB are read with the **standard library only**
(`html.parser`, `zipfile`, `xml.etree`). No BeautifulSoup, `lxml`, or
`python-docx`: §9's constraint held through M5.

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
  Cost estimation is out of scope. `-v` is accepted both before the subcommand and
  on `index`/`query` themselves.
  Third-party recovery chatter is suppressed below `ERROR` unless `-v` is given:
  `pypdf` logs a warning for each quirk in a malformed file and then extracts the
  page anyway, so those lines are not actionable and they interleave with the
  progress bar. A page we genuinely cannot read is reported by us, not by pypdf.
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

## 11. Roadmap after the PoC

The PoC is complete and verified (§14). This section sequences the items deferred
in §1.1, plus the improvements the implementation itself surfaced, into
milestones. §11.9 maps every deferred item to the milestone that carries it, so
nothing from the original design is silently dropped.

**Ordering principle.** Correctness before coverage, coverage before tuning, and
measurement before any retrieval experiment:

1. **Correctness first (M4).** The index can currently go *wrong* rather than
   merely incomplete. Nothing records which embedding model wrote a store
   (verified: `vectorstore.py` stores no model identity), so swapping
   `embedding_model` for another of the **same dimension** — `text-embedding-3-small`
   for `ada-002`, both 1536 — mixes incompatible vectors and returns meaningless
   neighbours with no error and no warning. A *different* dimension at least fails
   loudly (`Collection expecting embedding with dimension of 3, got 1536`), which
   is the same bug getting lucky. A confidently cited wrong answer is the worst
   failure this tool has.
2. **Coverage next (M5).** A survey of the live library found **1,022 attachments,
   of which only 670 are PDFs**: 205 are HTML snapshots and 105 items have *nothing
   but* a snapshot, so they are invisible to `query` today (§11.2 has the full
   breakdown). Better retrieval over a corpus that excludes those items is worth
   less than indexing them.
3. **Then measurement (M6), then tuning (M7).** Rerankers, hybrid search, and
   `top_k` changes are all *hypotheses*. Without a scored golden set we would be
   choosing them by how good one answer looked, which is how retrieval
   regressions get shipped.

Sizes are rough: **S** ≈ a day or less, **M** ≈ a few days, **L** ≈ longer.

| # | Milestone | Theme | Version | Size |
|---|---|---|---|---|
| 4 | Store integrity and true reindexing | correctness | 0.2.0 | ✅ **done** |
| 5 | Cover more of the library (HTML, text, DOCX/ODT) | coverage | 0.3.0 | ✅ **done** |
| 6 | Measurement before tuning | evaluation | 0.4.0 | ✅ **done** |
| 7 | Retrieval quality | tuning | 0.5.0 | 🚧 windowing done (§11.4.1); rest M |
| 8 | Scale and cost | performance | 0.6.0 | S–M |
| 9 | Daily use | ergonomics | 0.7.0 | M |
| 10 | Reach: group libraries, other stores, MCP | breadth | 0.8.0 | M |
| 11 | Engineering hygiene | infrastructure | any time | S |

The cheapest items worth doing first, regardless of milestone: the store
fingerprint (§11.1), naming *what* was skipped (§11.2), windowed retrieval and
metadata filters (§11.4), and `status` (§11.6). Each is small and independently
useful.

### 11.1 Milestone 4 — Store integrity and true reindexing (0.2.0) — ✅ done

**Status: implemented and verified live** (§14.6). Before it, the attachment key
was the only incremental criterion (§7.3): a replaced PDF was never re-read, a
deleted item never removed, and a changed embedding model never noticed.

**Goal:** make the index trustworthy when either the library or the settings
change.

- **Store fingerprint** — see §11.1.1 for where it lives and why. **The single
  highest-value item in the roadmap per hour of work:** it converts a silent
  wrong-answer mode into a clear error.
- **`index --force`.** Drop the collection and re-embed from `cache/`, with no
  re-download. `--force --refetch` also clears the cache.
- **`index --repair`.** Re-embed only the chunks a superseded model wrote, using
  the text already in the store — no download and no re-chunking, so chunk
  boundaries are preserved exactly. This is the remedy for an edited
  `embedding_model`; a changed `chunk_size` needs `--force` instead, because the
  boundaries themselves are what changed. When the new model has a different vector
  width the collection cannot be updated in place (Chroma fixes a collection's
  dimension at first write), so the store is rebuilt from its own text.
- **Version-aware incremental indexing.** Each attachment's Zotero `version` is
  stored on its chunks and compared on the next run; a newer version re-embeds that
  attachment after deleting its old chunks.

  *Implemented differently from the plan, deliberately.* The plan was to record
  `zot.last_modified_version()` in `config.yaml` and query `zot.items(since=…)`.
  That bookkeeping turned out to be unnecessary: a run must fetch the whole
  collection anyway to know its membership, so comparing per-attachment versions
  from that fetch is both exact and stateless. `since` would only reduce API
  traffic, which is negligible next to embedding cost — and a version recorded in
  `config.yaml` would have been another user-editable file claiming to describe
  the store, which is the mistake §11.1.1 exists to avoid.
- **Deletion sync and `--prune`, unified.** One set difference — stored attachment
  keys minus planned ones — covers deletion, an item moved out of the collection,
  and an attachment that stopped being a supported type. `zot.deleted()` reports
  only the first of those, so it was not needed. Orphans are *reported* by every
  run and deleted only by `--prune`: silently discarding data during an ordinary
  index run would be a poor default.

**Acceptance:** changing `embedding_model` and running `query` is refused with a
message naming that setting; replacing one `Writing` PDF in Zotero re-embeds
exactly that attachment; deleting an item removes its chunks and it stops being
cited; `--force` on `Writing` reproduces 146 chunks with zero downloads.

#### 11.1.1 Where the embedding-model record must live

`config.yaml` cannot be the record. It states what the user **wants**, is meant to
be edited, and is the very thing that changes when this bug bites. The store must
carry what was actually **used**, so the check becomes *intent vs. fact* rather
than config vs. itself.

Three layers, each verified against `chromadb` 1.5.9:

1. **Chroma collection metadata — the authoritative record.** Set at
   `get_or_create_collection(name, metadata={...})`, readable as
   `collection.metadata`, **persisted across a client reopen**, and updatable later
   via `collection.modify(metadata=...)` (needed for the migration below). Values
   must be scalars — a nested dict raises `ValueError` — so use flat dotted keys
   (`zotero_rag.embedding_model`, `zotero_rag.embedding_dim`,
   `zotero_rag.endpoint_host`, `zotero_rag.chunk_size`, `zotero_rag.chunk_overlap`,
   `zotero_rag.schema`) or one JSON string. This record lives inside `chroma/`,
   travels with the store if it is copied, and is not a file anyone hand-edits.
2. **Per-chunk `embedding_model` in the chunk metadata — detects a *mixed* store
   and enables repair instead of rebuild.** A single global fingerprint only tells
   you the store is suspect; a per-chunk field tells you *which* vectors are stale.
   Verified: `get(where={"embedding_model": {"$ne": current}})` returns exactly the
   stale chunk ids, and the distinct models present can be enumerated. That is what
   makes `index --repair` (re-embed only the stale chunks, from `cache/`, no
   re-download) possible — far better than forcing a full rebuild after an
   accidental edit. Cost is one short string per chunk.
3. **The vector dimension, derived rather than recorded.** `collection.get(limit=1,
   include=["embeddings"])` yields a real vector, so its length is a *fact* that
   cannot drift from the data or be forged by editing a file. Use it to cross-check
   layer 1, and to reconstruct a fingerprint for stores written before this
   milestone.

**Optional fourth layer — a canary probe, for drift a model name cannot express.**
An OpenRouter slug is not a version: the same `openai/text-embedding-3-small` may
be served by a different provider or revision over time, and no recorded string
would notice. Embedding one fixed canary sentence at index time and storing that
vector (as a reserved id, or in the collection metadata) allows a later
`--verify` to re-embed it and compare — cosine similarity, not equality, since
embeddings are not bit-stable across providers; below ~0.999 means the endpoint is
no longer producing the vectors this store was built with. Cost is one embedding
call, so make it opt-in rather than part of every query.

**Behaviour on mismatch**, distinguishing corruption from mere inconsistency:

| Change | Response |
|---|---|
| `embedding_model` or dimension differs | **Refuse** both `index` and `query`, name the setting, offer `--repair` or `--force` |
| `chunk_size` / `chunk_overlap` differ | **Warn and continue** — the store is inconsistent, not wrong |
| Endpoint host differs | Warn; suggest `--verify` |
| No fingerprint at all (pre-M4 store) | Adopt: derive the dimension from a stored vector, record the config's current model as fact, warn once that an `embedding_model` edited since indexing cannot be detected retroactively |

The adoption path matters because the existing `Writing` and `Networking` stores
have no fingerprint, and forcing a re-embed of 1,296 chunks to gain a guard would
be a poor trade.

### 11.2 Milestone 5 — Cover more of the library (0.3.0) — ✅ done

**Status: implemented and verified live** (§14.7), except OCR and linked-URL
fetching, both deliberately deferred — see the end of this section.

**Goal:** index what is actually in the collections, not only the PDFs.

**Measured, not guessed.** A survey of the whole live library — 1,022 attachments
across 801 items with at least one attachment — says which parsers pay off:

| Content type | Attachments | Status today |
|---|---|---|
| `application/pdf` | 670 | indexed |
| `text/html` (snapshots) | 205 | **skipped** |
| *(none)* — almost all `linked_url` | 132 | skipped |
| `text/plain` | 10 | **skipped** |
| `application/vnd.oasis.opendocument.text` (ODT) | 1 | skipped |
| `…wordprocessingml.document` (DOCX) | 1 | skipped |
| `image/png`, `image/jpeg` | 2 | skipped (OCR territory) |
| `text/csv` | 1 | skipped |

Link modes: 468 `imported_url`, 388 `imported_file`, 166 `linked_url`.

Per *item*, which is what actually determines coverage:

| Item has | Items | Consequence |
|---|---|---|
| PDF only | 555 | already covered |
| **HTML snapshot only** | **105** | **completely invisible to `query` today** |
| PDF **and** HTML snapshot | 100 | snapshot would *duplicate* the PDF |
| no PDF and no snapshot (`(none)`) | 35 | linked URLs |
| `text/plain` only | 5 | invisible |
| ODT only | 1 | invisible |

**The decisive finding: of the 205 snapshots, ~100 sit on items that already have
a PDF.** Ingesting snapshots naively would add roughly a hundred near-duplicate
copies of documents already indexed — they would compete with the PDF chunks in
retrieval and could yield two citations for one source. So format support needs a
**per-item preference order** rather than "index everything readable":

# TO FIX: Include markdown with precedence over txt
```yaml
  # First format found for an item wins; the rest are recorded as skipped.
  attachment_preference: [pdf, docx, odt, html, txt]
  index_all_attachments: false   # true = index every readable attachment
```

With that in place the payoff is **+105 items (+13% of items with attachments)**
newly searchable and no duplication, versus +205 attachments and ~100 redundant
ones without it.

Scope, ordered by measured payoff:

- **Say what was skipped (S).** The dry run and the run summary should name content
  types and titles per skip reason. `10  unsupported content type` gives the user
  nothing to act on — and this survey is exactly what a user cannot currently
  produce for themselves.
- **HTML snapshots — 205 attachments, 105 otherwise-invisible items.** The single
  biggest coverage win in the project. A stdlib `html.parser` extractor (drop
  `script`/`style`/`nav`, collapse whitespace, keep heading structure for the
  locator) adds no dependency.
- **Plain text and Markdown — 10 attachments, and free.** `text/plain`,
  `text/markdown`, and `text/csv` are all read-then-split with no parser at all;
  they share one code path with the extractor above.
- **DOCX and ODT — 1 each here, but cheap and symmetric.** Both are ZIP + XML:
  DOCX is `word/document.xml` (`w:p` → `w:t` nodes), ODT is `content.xml`
  (`text:p`). Roughly thirty lines each with `zipfile` + `xml.etree`, no dependency
  — `python-docx` would pull in `lxml` for one document in this library. Worth
  doing as a pair for other people's libraries; honestly not worth doing *first*
  for this one.
- **Zotero notes** (`note` items): index the note body with the parent item's
  metadata.
- **EPUB**: `zipfile` plus the same HTML extractor, chapters as the locator. None
  present in this library, so lowest priority of the file formats.
- **`linked_url` attachments — 166, opt-in only** (`ingest_linked_urls: false`).
  Fetching arbitrary URLs is a different risk class from reading files Zotero
  already stores, and these are the least reliable sources in the library.
- **OCR, optional and external.** Only via an external binary (`ocrmypdf` /
  `tesseract`) detected on `PATH`, controlled by `ocr: auto|never|always`. A Python
  ML OCR stack would violate §9, and that constraint stands. Two images in the
  whole library, so this is about scanned PDFs, not about the `image/*` rows above.
  Decision: OCR is not in the near future priorities.

**The one hard design consequence: `page` must become a locator.** §7.2 stores a
1-based PDF page, and §8.2 renders it as `p. 4`. HTML, Markdown, DOCX, and notes
have no pages, so the chunk gains a `locator_kind` (`page` | `section` |
`paragraph` | `chunk`) beside the value, and citations render accordingly — `p. 4`,
`§ Introduction`, `¶ 12`, `chunk 3`. Retrofitting this after several formats exist
would be worse than doing it once, up front, so it belongs at the start of M5 and
not at the end. Existing chunks default to `locator_kind: page`, which is what they
are.

**Acceptance:** the summary names each skipped content type with counts; enabling
snapshot ingestion on a collection containing snapshot-only items makes them
answerable and cites them with a section locator; an item holding both a PDF and a
snapshot yields chunks from the PDF only under the default preference; the `Writing`
regression (146 chunks, `p. N` citations) is unchanged. **All verified — §14.7.**

#### 11.2.1 Deferred from M5, and why

- **OCR.** Not implemented. Neither `tesseract` nor `ocrmypdf` is present on the
  development machine, so the invocation path could be written but not *verified*,
  and shipping an unexercised subprocess call that runs over a user's whole library
  is worse than shipping nothing. The design stands as written above — external
  binary only, `ocr: auto|never|always` — and it needs a machine with the binary
  installed to be finished honestly.
- **Linked-URL fetching** (166 attachments). Not implemented. Everything else in
  M5 reads bytes Zotero already stores; fetching arbitrary URLs is a different risk
  class — it makes the tool a web client, with redirects, paywalls, dead links and
  whatever a URL resolves to today. It stays behind the planned
  `ingest_linked_urls: false` and wants its own milestone rather than a corner of
  this one. This is not a feature for the near future.

Both remain listed in §11.9 rather than being quietly dropped.

### 11.3 Milestone 6 — Measurement before tuning (0.4.0) — ✅ done

**Status: implemented and verified live** (§14.8). Cost is the one listed item not
delivered; it needs M8's pricing table and is recorded in §11.9 rather than
quietly dropped.

**Goal:** stop judging retrieval by eye. §12.1 already defines three questions
with expected points — make them executable.

- **`questions.yaml`** in the index folder: question, expected points, and
  optionally the item keys a correct answer must cite. Built as specified, with
  one addition: a question may also carry a prose `answer:`. A reference answer
  generated alongside a question is prose, not a bullet list, and forcing it into
  points would discard the more precise statement — so points and prose are both
  accepted, and both are handed to the judge. Points are what coverage counts.
- **`zotero-rag eval [--json]`**: built as specified (§3.5). Reports retrieval hit
  rate, point coverage judged by an LLM, tokens and latency — a table plus one
  summary line.
- **Comparable runs:** temperature 0, and each run records the settings that
  affect it. See the caveat on determinism below: temperature 0 is necessary but
  not sufficient.
- Reuse `ragbench`'s scoring shape (§13) rather than inventing one. Done — the
  5-level ordinal grade and its scores are `ragbench.models.GradeLabel`, and
  `--assess-context` reuses its `ContextVerdict` labels. **One deliberate
  divergence:** ragbench defaults a failed judge call to `wrong`, which reports a
  grading outage as a retrieval regression. A seventh label, `error`, scores zero
  but is counted separately, so a flaky provider cannot masquerade as a worse index.

**Three metrics, measuring deliberately different things.** Hit rate is
mechanical — did any chunk of an expected item reach `top_k` — so it cannot drift
with a model revision, and it separates retrieval failure from generation failure.
The citation rate is stricter and is what §12.7 actually asks for: a passage that
reached `top_k` but went uncited did not support the answer. Point coverage needs
the LLM, because "uno es tan bueno como su último paper" and "previous success
does not guarantee future success" are the same point and no string match sees that.

**Acceptance:** `eval` on `Writing` reproduces the §12 judgement — all three
questions covered, at least one correct citation each — and a deliberately poor
setting (`top_k: 1`) scores measurably worse.

**Verdict: half of the first clause failed, and that is the milestone working.**
The citation half holds — 100% of `Writing` questions cite a real `Writing` item.
The "all three questions covered" half does not: measured coverage is 22–40% and
one question grades `wrong` outright. §12.7's "covering most of the expected
points" was an eyeball judgement, and the first thing measurement did was
contradict it. Rather than tune the judge until it agreed, the number stands —
§14.8 has the detail, including why the `publicacion` question genuinely fails.

The `top_k: 1` clause passed, but only on the larger set: see §11.3.1.

#### 11.3.1 What measurement immediately said about measurement

Two findings that only appear once the harness exists, both of which change how
§11.4 must be run:

- **Three questions cannot measure anything.** On `Writing`, `top_k: 1` scored
  *better* than `top_k: 5` (mean 2.00 vs 1.33), the opposite of the acceptance
  criterion's prediction, and two identical `top_k: 5` runs disagreed by 14
  coverage points. On the 21-question `Networking` set the same comparison is
  unambiguous — mean 0.76 at `top_k: 1` against 1.81 at 5 and 2.38 at 10 — and
  the repeat run reproduced every mechanical metric exactly. A 3-question set is
  a smoke test; a tuning decision needs the larger one.
- **Temperature 0 does not make an LLM judge deterministic.** Repeat runs
  reproduced the *grade labels* and every mechanical metric bit-for-bit, while
  point coverage moved by 2–3 points on 21 questions and up to 14 on 3. So the
  ordinal grade and the hit/citation rates are the trustworthy comparators, and a
  coverage difference smaller than a few points is noise. The mechanical metrics
  are the ones worth regressing against, which is the argument for having them.

#### 11.3.2 `generate-questions` (0.4.1) — ✅ done

**Status: implemented and verified live** (§14.9).

**The problem M6 left behind.** Evaluation is executable, but a golden set still
has to be written by hand, and §11.3.1 showed how much that costs: the 3-question
`Writing` set cannot resolve a tuning decision, and the 21-question `Networking`
set that can took an afternoon of reading. The harness is therefore bounded not by
code but by how many questions someone is willing to write — and a user with a
50-paper collection will write none.

**`zotero-rag generate-questions`** (§3.6) writes a starting `questions.yaml` from
the documents already indexed: one LLM call per document, yielding
`questions_per_document` questions each with a reference answer and expected points.

Two design decisions carry the value:

- **Generate from the vector store, not from `cache/`.** The cache is the obvious
  source — the PDFs are sitting there — but it is the wrong one. A question written
  from text that is not *in the index* can never be answered from the index, so it
  would sit in the golden set as a permanent false failure, silently depressing
  every score and every comparison built on it. Generating from the store inverts
  that: every question is answerable in principle, so a failure means retrieval or
  generation genuinely failed. It also needs no downloads and no re-parsing, and it
  still works after `--force --refetch` has emptied the cache.
- **`expected_items` is exact, because the generator knows where the text came
  from.** This is the property that separated the two hand-written sets: the
  `Networking` set could name one source article per question and so measures
  retrieval discrimination, while `Writing` could only check that retrieval stayed
  inside the collection (§12.1, §12.2). Generated sets get the strong version for
  free, which is the single best reason to prefer them as a starting point.

Supporting behaviour, all of it following existing conventions: `--dry-run` lists
the documents with character counts and an estimated prompt-token cost and writes
nothing; `--out` and `--force` exist because silently overwriting a curated set
would be the worst thing this command could do, and the check happens *before* any
model call rather than after; `--only-item` and `--limit` narrow a trial run; a
collection over 30 documents asks for confirmation, as a whole-library `index`
does; one document whose generation fails is reported and skipped rather than
losing the rest of the run (§10).

**A long paper is sampled head-and-tail, not truncated.** `document_char_budget`
defaults to 30,000 characters, and 14 of `Networking`'s 16 documents exceed it. A
plain truncation would hand the model the abstract and introduction and drop the
results and conclusions — precisely the "main points" the questions are supposed to
be about — so 70% of the budget comes from the head and 30% from the tail, with the
gap marked and the prompt told not to ask about the omitted middle.

**The written file is read back through `load_questions` before the command
returns.** Emitting YAML that `eval` cannot parse is the obvious failure mode here:
reference answers are prose full of colons, quotes and accents, every one a YAML
hazard, and hand-quoting them is exactly what went wrong while writing
`Networking.yaml` by hand. Serialising with PyYAML and then re-parsing makes
"generation cannot produce something evaluation cannot read" a checked guarantee
rather than an intention.

**Acceptance:** `generate-questions` on `Networking` writes a set that `eval` runs
with no hand-editing, whose scores are comparable to the hand-written set, and
`--dry-run` estimates the cost without spending it. **All verified — §14.9.**

**What a generated set is not, and the file says so itself.** Every question comes
from a single document, so the set tests whether retrieval can find the document a
question was written from. That is a real regression harness and it is not the same
as measuring what you want to know: it under-tests synthesis across several papers,
and it structurally cannot ask a question the corpus fails to answer — so it can
never produce the §12.8 "I don't know" case. The header comment written into every
generated file says all of this and tells the user to edit it, because a draft that
looks finished is worse than one that admits it isn't.

**Deferred, deliberately.** A `--merge` mode that adds questions only for documents
not already covered would make the command usable incrementally as a collection
grows; today a second run replaces the file. Worth having, not worth blocking on.

### 11.4 Milestone 7 — Retrieval quality (0.5.0)

**Depends on M6.** Each item below is a hypothesis to be accepted or rejected by
`eval`, cheapest first.

- **Windowed retrieval (S) — ✅ built, ❌ rejected as a default.** Implemented as
  `retrieval_window` / `--window` (§3.4) and measured; the mechanism works and the
  numbers say not to switch it on. §11.4.1 has the frontier and the reasoning. This
  is the first hypothesis this section put to `eval`, and the answer was no — which
  is what §11.3 was built for.
- **Retrieval diversity (S) — surfaced by §11.4.1, not in the original list.** A
  per-document cap, or maximal-marginal-relevance over the candidates. At `top_k:
  30` one question came back with all thirty chunks from a single document, while
  the paper it was actually about never appeared at any `top_k`. Nothing else on
  this list addresses that, and it is the cheapest item here.
- **Query-time metadata filters (S).** `--year`, `--author`, `--collection`,
  `--doi` mapped to Chroma `where` clauses. The metadata is already stored (§7.2);
  `year` needs an integer companion field for range comparisons. §11.4.1's
  persistent miss is a candidate: `--author ALzaid` would have found it.
- **Hybrid retrieval — ✅ built, ✅ accepted, and now the default.** BM25 via
  `rank_bm25` fused with the vector hits by reciprocal rank fusion, behind
  `retrieval: vector|hybrid` (§3.4). It measured better on two independent question
  sets **for no extra tokens**, which is what separates it from windowing.
  §11.4.2 has the numbers.
- **Reranking without heavy dependencies — ✅ built, ✅ accepted, off by default.**
  Implemented with the "possible second provider" this plan mentioned rather than the
  LLM scorer: Cohere's hosted `rerank-v4.0-fast` behind `reranker: null|cohere`,
  retrieving `top_k × fanout` candidates first. It is the best result M7 has produced
  and it **reduces** token spend; the switch stays off only because it needs a
  credential most users will not have. §11.4.3 has the numbers.
- **Revisit the default `top_k` — M6 has now answered this, and the answer is
  yes.** On the 21-question `Networking` set every metric improves monotonically
  with `top_k`, and the step from 5 to 10 is larger than the noise floor by an
  order of magnitude:

  | `top_k` | mean score | point coverage | retrieval hit | expected item cited | tokens |
  |---|---|---|---|---|---|
  | 1 | 0.76 | 22.7% | 71% | 29% | 32.6k |
  | 5 (default) | 1.81 | 53.3% | 86% | 76% | 58.2k |
  | 5 (repeat) | 1.81 | 56.0% | 86% | 76% | 58.2k |
  | 10 | **2.38** | **67.9%** | **95%** | **81%** | 86.7k |

  So on a 1,186-chunk store over 16 documents, `top_k: 10` buys +0.57 mean score
  and +9 points of retrieval hit for ~1.5× the tokens. This is now a measured
  trade-off rather than taste — but it is one store, and it is the *first* thing
  §11.4 should re-run after any retrieval change, not a settled default. Raising
  the shipped default is left to M7, where windowed retrieval and reranking
  compete for the same context budget and should be compared against this table
  rather than against each other in isolation.

  Note the shape of the `top_k: 1` failure: 13 of 21 questions answered "I don't
  know" rather than guessing from one chunk. The §12.8 abstention behaviour holds
  up under starvation, which is the good half of a bad setting.

  §11.4.1 extends this to `top_k` 20 and 30 and finds the returns stop.

#### 11.4.1 Windowed retrieval, measured and rejected (0.4.2)

**Built as specified.** `expand_window()` in `querying.py` fetches each hit's ±N
neighbours in reading order, deduplicates where two hits' windows overlap, and
preserves the hit's similarity distance. Off by default (`retrieval_window: 0`),
overridable per run with `--window` on both `query` and `eval`, and recorded in the
eval report so a windowed run cannot be mistaken for a plain one.

Two implementation notes worth keeping:

- **Neighbours are taken across the whole attachment, not within the page.** This
  looks like a detail and is the opposite: `chunk_overlap` already stitches
  consecutive chunks *inside* a segment together, so a segment boundary is the one
  place in the store where no overlap exists at all. A window confined to one page
  would skip the only join that is genuinely unbridged.
- **A neighbour on another page becomes its own numbered source**, rather than
  being folded into the hit's. The text really is on that page, and citing where it
  sits beats attributing it to the page next door. `top_k` therefore still bounds
  the *hits*, while the source count may exceed it — which is why `QueryResult`
  now carries `chunks_retrieved` and `chunks_in_context` separately.

**The measurement.** All runs on the 21-question `Networking` set (§12.2), ordered
by the tokens spent answering — because every one of these levers buys quality with
context, so cost is the axis they have to be compared on:

| answer tokens | `top_k` | window | mean score | point coverage | hit | cited |
|---|---|---|---|---|---|---|
| 33.2k | 5 | 0 | 1.81 | 53.3–56.0% | 86% | 76% |
| 61.5k | 10 | 0 | 2.38 | 67.9% | 95% | 81% |
| 74.0k | 5 | **1** | 1.76 | 52.5% | 86% | 71% |
| 111.1k | 5 | **2** | 2.14 | 68.6% | 86% | 71% |
| 117.8k | 20 | 0 | 2.48 | 74.5% | 95% | 86% |
| 136.6k | 10 | **1** | 2.33 | 68.4% | 95% | 86% |
| 174.0k | 30 | 0 | 2.48 | 76.4% | 95% | 86% |
| 196.9k | 10 | **2** | **2.62** | 74.6% | 95% | 86% |

**Verdict: every windowed configuration is dominated by, or tied with, a
non-windowed one costing the same or less.**

- `window: 1` is simply bad — worse than its own baseline at both `top_k` values
  (1.76 vs 1.81; 2.33 vs 2.38) for 2.2× the tokens. It adds enough text to dilute
  and not enough to complete: `zheng-1` and `wang-2` both fell from `partial` to
  `wrong`, and the citation rate at `top_k: 5` dropped 76% → 71%.
- `window: 2` genuinely helps — `wang-1` `wrong` → 80% coverage, `cfr-rl-3` 8% →
  58% — but `top_k: 5, window: 2` (111k, 2.14) is beaten outright by `top_k: 10,
  window: 0` (61k, 2.38): better score, comparable coverage, 45% fewer tokens.
- The best single result *is* windowed — `top_k: 10, window: 2` at 2.62 — but its
  fair comparison is `top_k: 30, window: 0` at 174k: score 2.62 vs 2.48, coverage
  74.6% vs 76.4%. One metric each, with the coverage gap inside the noise band
  §11.3.1 measured, for 13% more spend. That is a tie, not a win.

**Why this is not surprising, in hindsight.** A neighbour is by construction in the
same attachment — therefore the same item — as its hit, so **windowing cannot change
retrieval hit rate at all**: it was 86% at `window` 0, 1 and 2 alike, and 95% at
every `top_k ≥ 10` regardless of window. Raising `top_k` brings in *new documents*;
widening the window only elaborates documents already found. On a corpus where the
binding constraint is still finding the right paper, the lever that cannot address
that loses to the one that can.

**Kept, not enabled.** The feature stays: it is small, config-gated, tested, and the
mechanism is sound. A corpus with much smaller chunks, or long documents whose
argument genuinely spans many chunks, could invert this — and the point of §11.3 is
that anyone can now find out for their own collection with one `eval --window`.

**A second result, free from the same runs: `top_k` returns stop.** 5 → 10 is worth
+0.57 mean score; 10 → 20 is +0.10; 20 → 30 is **+0.00** (coverage +1.9 points,
inside noise). So the useful range on this store ends around 10–20, and anything
past 20 is pure spend.

**The shipped default became `top_k: 10` in 0.4.4.** It was held at 5 through 0.4.3
on the argument that a reranker might change the arithmetic — but by then 5 → 10 had
improved *every* metric in *every* configuration measured: both retrieval modes, both
question sets, windowed and not. Holding a default at a value that never won any
comparison, on the strength of a feature that does not exist yet, is not caution but
inertia. The trade is explicit and belongs in the release note rather than in a
footnote: **roughly 1.9× the tokens per query** for +0.57 mean score and +9 points of
retrieval hit rate. `top_k: 5` remains one line of config for anyone who would rather
have the cheaper query, and when reranking does arrive it will be measured against
this default rather than the old one.

**A changed default reaches new index folders only, and that is worth knowing before
wondering why nothing improved.** `init` copies the packaged `config.yaml` *by value*
(§3.2), so a folder created before 0.4.4 holds `top_k: 5` as an explicit setting, and
by §5.1's precedence an explicit local value beats the packaged default. Verified: an
existing `Networking` folder still resolved `top_k=5` after the change, and scored
1.71 — its old numbers — while a freshly initialised folder resolved 10.

Two consequences, one good and one to act on:

- Nobody gets a surprise bill from upgrading. The token cost arrives only when a user
  edits the file or creates a new folder, which is the right way round for a change
  that doubles per-query spend.
- **Existing folders need one edit to benefit**: set `top_k: 10` in their
  `config.yaml`. Nothing tells them so, which is precisely the gap §11.8's *config
  schema version and migration* item describes. The same run also showed the other
  half of that gap working in our favour: `retrieval` was *absent* from the old file,
  so it correctly inherited the new `hybrid` default. Keys present are pinned; keys
  absent are inherited — a reasonable rule that no user has been told about, and the
  case for a `status` command (§11.6) reporting effective settings rather than
  configured ones.

**The most useful thing these runs found is a question that no amount of context
fixes.** `jellyfish-3` misses at `top_k` 5, 10, 20 *and* 30, and at every window.
Its text is "¿Qué metodologías de evaluación utilizaron los autores para validar sus
propuestas...?" — which names nothing from the Jellyfish paper, so all 30 slots fill
with the *systematic literature review* on evaluation methodology, a paper the
question describes better than its own source does. Two things follow, and neither
is about context size:

- **Retrieval has no diversity constraint.** Thirty chunks came back and every one
  was from a single document. A per-document cap or maximal-marginal-relevance
  belongs on this list, and would have cost nothing here.
- **"The authors" is under-specified for a corpus-wide search.** The question was
  written while holding one paper, and reads as ambiguous once 16 are in scope. The
  §11.3.2 generator prompt already forbids exactly this phrasing in *generated*
  sets; the hand-written set of §12.2 has the flaw the generator was told to avoid.
  Worth fixing there, and worth remembering when writing questions by hand.

#### 11.4.2 Hybrid retrieval, measured and adopted (0.4.3)

**Built as specified**, in a new `retrieval.py`: BM25 over every stored chunk,
reciprocal rank fusion with the vector ranking, behind `retrieval: vector|hybrid`
plus `--retrieval` on `query` and `eval`. `rank_bm25` is the one new dependency;
numpy came with `chromadb` already, so §9 holds.

Four decisions that shaped the result:

- **Fusion is rank-based, not score-based.** A cosine distance and a BM25 score
  share no scale, and normalising them — min-max over the candidates, say — makes the
  merged order depend on how extreme the *worst* candidate happened to be. Positions
  are comparable; magnitudes are not.
- **Each retriever runs deeper than the final cut** (`max(4 × top_k, 20)`). Fusing
  two `top_k`-long lists gives the merge almost nothing to work with; the case worth
  catching is a chunk ranked 14th by vectors and 2nd by BM25.
- **Tokenizing folds accents and splits on punctuation.** Folding is what lets a
  Spanish question match text with different diacritics — without it half the §12.2
  questions match nothing. Splitting means `CFR-RL` → `["cfr", "rl"]`, so the query
  and the paper agree however either punctuates the name. No stopword list: BM25's
  IDF already discounts ubiquitous words, and a fixed list would have to pick a
  language, which is wrong for a corpus queried in Spanish and written in English.
- **The keyword index is built once per process**, not per question. It is a
  tokenize pass over the whole store — 0.21s for 1,186 chunks — and `eval` over 48
  questions must not pay that 48 times.

**The measurement.** Two independent question sets, and the token columns are the
point: fusion changes *which* chunks reach the model, not how many, so it is
essentially free.

Hand-written set, 21 questions (§12.2):

| mode | `top_k` | mean score | point coverage | hit | cited | answer tokens |
|---|---|---|---|---|---|---|
| vector | 5 | 1.81 | 53.3% | 86% | 76% | 33.3k |
| **hybrid** | 5 | 1.81 | 55.8% | 86% | 81% | 33.7k |
| vector | 10 | 2.38 | 67.9% | 95% | 81% | 61.5k |
| **hybrid** | 10 | 2.33 | 72.1% | 95% | **95%** | 62.9k |
| **hybrid** (repeat) | 10 | 2.52 | 72.3% | 95% | **95%** | 62.8k |

Generated set, 48 questions (§11.3.2) — an independent sample:

| mode | `top_k` | mean score | point coverage | hit | cited | answer tokens |
|---|---|---|---|---|---|---|
| vector | 5 | 2.08 | 61.6% | 81% | 73% | 72.5k |
| **hybrid** | 5 | **2.38** | **67.7%** | **90%** | **77%** | 71.5k |

**Verdict: accepted, and made the default.** The evidence, in the order it should be
trusted (§11.3.1: mechanical metrics first, then coverage, then the ordinal score):

- **Citation rate** — mechanical, exactly reproducible — rose 81% → **95%** at
  `top_k: 10`, identical across both hybrid runs. Twenty of twenty-one answers cite
  the paper they should.
- **Retrieval hit rate** rose 81% → **90%** on the 48-question set: roughly four more
  questions found their source paper. On the hand-written set it did not move (86%,
  95%), for a reason given below.
- **Point coverage** rose on both sets by 4–6 points, above the ±2.7 noise floor, and
  reproduced to within 0.2 points across the two hybrid runs.
- **Mean score** rose on the generated set (2.08 → 2.38) and was flat on the
  hand-written one — the apparent 2.38 → 2.33 dip was judge noise, since the repeat
  gave 2.52.
- **Cost: nil.** 33.3k → 33.7k and 61.5k → 62.9k tokens, and the 48-question run was
  actually 1k *cheaper*. Compare windowing, which bought its gains at 2–3× (§11.4.1).

**Why hit rate improved on the generated set but not the hand-written one.** Generated
questions were written while reading their source paper, so they echo its vocabulary —
`PEMA`, `PEDL`, `Ee-Routing`, `DMCF`. That is precisely the case BM25 is for. The
hand-written questions are paraphrases, which is exactly where BM25 has nothing to
contribute and dense retrieval already works. The two sets did not disagree; they
measured different things, and having both is what made that visible.

**Hybrid wants `top_k ≥ 10`, and here is the failure that shows why.** At `top_k: 5`,
fusion gives roughly half the slots to each retriever, so a run keeps only two or
three vector hits. `sdn-review-3` — "¿cuáles son los cuatro beneficios principales que
SDN ofrece?" — graded `good` under vector and **`no answer`** under hybrid at
`top_k: 5`: the right item was still retrieved, but the chunk actually listing the four
benefits was displaced by keyword matches, and the model correctly declined rather than
invent them. At `top_k: 10` it is `good` again in both hybrid runs. The abstention is
the system behaving well (§12.8) around a retrieval setting that is too tight; no run
ever under-filled `top_k` or returned nothing.

**Default changed to `hybrid`** (0.4.3). Enabling hybrid imposes no token cost, so no
user gets a surprise bill from upgrading — only ~0.2s per 1,000 chunks of start-up.

**`top_k` followed in 0.4.4** (§11.4.1), which makes the shipped defaults exactly the
best configuration measured: **`top_k: 10` with hybrid** — mean 2.33–2.52, coverage
72%, hit 95%, cited 95%, against the PoC default's 1.81 / 53% / 86% / 76%. That is
where the two levers differ: hybrid supplied its share of the gain for free, while
`top_k` had to be paid for in tokens.

**Deferred: persisting the keyword index.** It is rebuilt per process, which is 0.21s
at 1,186 chunks but extrapolates to ~9s on a 50,000-chunk store — paid on every
one-shot `query`. That belongs with M8's scale work, not here, and `retrieval: vector`
is the escape hatch until then.

**What hybrid did not fix.** `jellyfish-3` still misses at every `top_k` and in both
modes, exactly as §11.4.1 predicted: its question contains no distinctive term, so
there is nothing for BM25 to match either. Two of this section's remaining items —
retrieval diversity and metadata filters — are still the candidates for it.

#### 11.4.3 Reranking, measured and adopted as the recommendation (0.4.5)

**Built with a hosted cross-encoder**, in `reranking.py`: Cohere's
`rerank-v4.0-fast` behind `reranker: null|cohere`, plus `--reranker` on `query` and
`eval`. A local cross-encoder would mean `sentence-transformers` and PyTorch, which
§9 forbids outright; the hosted API needs **no new dependency at all**, because
`httpx` is already required (§14.7) and reranking is one POST.

Retrieval ranks by *similarity*, a reranker by *relevance to this question*. The
distinction is not cosmetic: an embedding is computed without knowing the question, so
it cannot tell a passage that is merely on the subject from the one that answers what
was asked. A cross-encoder reads both together.

**The stage order is fixed and deliberate: retrieve wide → rerank → window.**

- Reranking *before* the cut is what lets it discard rather than merely reorder. A
  reranker handed exactly `top_k` candidates changes nothing about which passages the
  model sees, and the ordering of a small context matters far less than its contents.
  Hence `reranker_fanout: 4` — `top_k: 10` retrieves 40 and keeps 10.
- Windowing *after* is what keeps a window's neighbours as context for a chosen hit
  rather than rivals competing to be chosen. Reranking them would let a low-scoring
  neighbour be dropped, which defeats the point of asking for it.
- It composes with `retrieval` untouched: hybrid fuses its candidates first, then the
  reranker sees the fused list. Verified live: `hybrid: 80 vector + 80 keyword
  candidate(s) fused to 20` followed by `reranked 20 candidate(s) down to 5`.

**The measurement**, 21-question `Networking` set, best rows in bold:

| `top_k` | retrieval | reranker | mean score | coverage | hit | cited | LLM tokens |
|---|---|---|---|---|---|---|---|
| 5 | vector | — | 1.81 | 53.3% | 86% | 76% | 33.3k |
| 5 | hybrid | — | 1.81 | 55.8% | 86% | 81% | 33.7k |
| **5** | **hybrid** | **cohere** | **2.38** | **73.8%** | **95%** | **95%** | **33.9k** |
| 10 | vector | — | 2.38 | 67.9% | 95% | 81% | 61.5k |
| 10 | hybrid | — | 2.33–2.52 | 72.1–72.3% | 95% | 95% | 62.9k |
| **10** | **hybrid** | **cohere** | **2.62** | **75.5%** | **95%** | **95%** | **63.4k** |

**Verdict: accepted, and it is the first change that buys quality while *reducing*
context.** Two readings, both worth stating:

- **At `top_k: 5`, reranking matches or beats `top_k: 10` without it, on half the
  LLM tokens.** 2.38 vs 2.38 on score, and *better* on coverage (73.8% vs 67.9–72.3%),
  with hit rate rising 86% → 95% and citation 81% → 95%. This is the classic reranker
  payoff — retrieve wide, send narrow — and it is the configuration to prefer.
- **At `top_k: 10`, reranking gives the best result the project has recorded**: 2.62
  mean, 75.5% coverage. The previous best equal to it was `top_k: 10, window: 2` at
  196.9k tokens (§11.4.1); this reaches it on 63.4k, a third of the context.
- **Reranking raised hit rate at `top_k: 5` from 86% to 95%**, which windowing could
  not do at all and hybrid could not do on this set. The reason is structural and the
  mirror of §11.4.1's: the reranker is choosing from 20 candidates rather than 5, so
  papers that were retrieved-but-cut now survive. It changes *which* documents reach
  the model, which is the only kind of change that moves this metric.

**A cost caveat that the token column does not show, and must not be read past.**
Those are **LLM tokens only**. Reranking adds one Cohere request per question — 20 to
40 documents each — billed by a second provider in search units, and our accounting
(§10) counts none of it. So "half the tokens" is a claim about *context*, not about
*money*: at these volumes the LLM saving is small in absolute terms and the rerank
charge is not obviously smaller. Settling that needs M8's pricing table, and until
then the honest claim is quality-per-context-token, not quality-per-dollar. Latency
is measurable now and modest: ~1s added to a single query.

**Off by default, and this is not timidity.** `reranker: cohere` requires
`COHERE_API_KEY`. Defaulting it on would make `query` fail outright for every user who
has only the OpenRouter and Zotero credentials that §5.2 asks for — trading a working
tool for a better one they cannot run. So the default is `null`, the credential is
asserted at build time rather than mid-run, and the recommendation lives in the
documentation: **if you have a Cohere key, `top_k: 5` with `reranker: cohere` is both
better and cheaper in context than the shipped default.**

**Failure policy, split by cause.** A 4xx — unknown model, rejected key — raises
`RerankError` and stops the run, because it will fail identically for every remaining
question and a hundred silent degradations is the worse outcome. Anything else
(timeout, 5xx, a 200 with no usable ranking) logs a warning and falls back to the
retrieved order, but records `reranked: false`, and `eval` prints *reranking applied
to only N of M questions*. A run that quietly half-reranked would otherwise be
indistinguishable from a reranked one while not being comparable to either.

**`jellyfish-3` still misses, now with 40 reranked candidates.** Three M7 levers have
now failed on it — deeper `top_k` (to 30), keyword fusion, and reranking over four
times the candidates. That is conclusive: the question names nothing from its own
paper (§11.4.1), so no amount of retrieval depth or re-scoring can find it. Metadata
filters, or fixing the question, are the only remaining answers, and it is the
strongest evidence in the PRD for the "under-specified question" failure mode.

### 11.5 Milestone 8 — Scale and cost (0.6.0)

- **Batched and concurrent embedding.** Embedding is serial per attachment today.
  Batch chunks per request and run a bounded worker pool across attachments, while
  keeping the per-attachment failure isolation of §10.
- **Cost estimation.** A pricing table in config (per 1M tokens, mirroring
  `ragbench`), surfaced under `-v` and in `--json` usage — and, most useful,
  `index --dry-run` estimating the spend *before* it happens.
- **Progress that scales.** ETA plus running token and cost counters, for runs
  much larger than the 13-PDF `Networking` case.

**Acceptance:** a `--force` re-index of `Networking` from cache is materially
faster in wall-clock than the serial path, and the dry-run estimate lands within
~10% of the tokens actually spent.

### 11.6 Milestone 9 — Daily use (0.7.0)

- **`zotero-rag status` (S).** Pinned collection, attachment and chunk counts, the
  models and fingerprint behind the store, cache size, library version, and when
  it was last indexed. The first thing wanted when returning to a folder after a
  month.
- **`chat` / REPL** with history, keeping the last N turns as conversational
  context while still citing every answer, plus `/sources` and `/topk`.
- **Streaming output** for `query`, so a long answer starts appearing immediately.
  `querying.stream_answer_question` already exists for `zotero-rag-ui` and returns
  the same `QueryResult`, so this is CLI wiring only.
- **Custom system prompt from a file** — `prompt_file` in config, `--prompt-file`
  to override (§8.1 currently fixes the prompt).
- **Friendlier CLI errors.** A stray positional such as `zotero-rag index
  Networking` should point at `--collection`; Typer's "Got unexpected extra
  argument" does not.
- **Cache management:** report cache size, and `--prune` files no longer
  referenced by the pinned collection.

### 11.7 Milestone 10 — Reach (0.8.0)

- **Group libraries.** `zotero_library_type: group` is already in the schema and
  entirely unused: plumb it through `build_client`, take a group id, and fix the
  citation URL builder (group item URLs differ from user ones). Cheaper than it
  looks.
- **Pinecone and other stores.** `vectorstore.py` is the only chromadb
  touchpoint, so this is a `VectorStore` protocol plus one new implementation,
  selected by `vector_db: chroma|pinecone`.
- **MCP server** exposing `query` as a tool, making an index usable from Claude
  Code and other MCP clients — plausibly the highest-leverage integration here,
  since it puts the citations where the writing happens.
- **Multiple collections per index folder** (a list of pinned collections),
  *if* one-folder-per-collection proves too rigid in practice. Worth deciding
  deliberately rather than drifting into.

### 11.8 Milestone 11 — Engineering hygiene (any time)

- **CI** running pytest, `ruff`, and `mypy` on push and pull request — deliberately
  deferred for now (§10), but the cost of adding it only rises.
- **Release process:** tag, changelog, `uv tool install` from a tag, and a version
  bump policy tied to these milestones.
- **Config schema version and migration**, so settings introduced by later
  milestones do not break folders created by earlier ones.

### 11.9 Where every deferred item landed

From §1.1 and the original §11 list:

| Deferred item | Milestone |
|---|---|
| `--force` clean-slate reindex | M4 |
| Version-aware incremental indexing | M4 |
| Deletion sync | M4 |
| OCR of scanned PDFs | M5 — **deferred**, §11.2.1 |
| HTML snapshot / EPUB / notes ingestion | M5 ✅ |
| Metadata filters at query time (year, author, tag) | M7 |
| Rerankers | M7 ✅ §11.4.3 |
| Hybrid / BM25 retrieval | M7 |
| Windowed retrieval | M7 |
| Cost estimation and pricing tables | M8 — also covers the `eval` cost column deferred from M6 |
| Batched / concurrent embedding | M8 |
| Custom system prompt from file | M9 |
| Interactive REPL with history | M9 |
| Streaming output | M9 |
| Group libraries | M10 |
| Pinecone and other vector stores | M10 |
| MCP server | M10 |
| CI (tests, `ruff`, `mypy`) | M11 |

Surfaced while building the PoC, not in the original list:

| Item | Milestone | Why |
|---|---|---|
| Store fingerprint / model-change guard | M4 | a changed `embedding_model` currently corrupts retrieval silently |
| `--prune` for items moved out of a collection | M4 | not a deletion, so `deleted()` never reports it |
| Name the skipped content types | M5 ✅ | `10 unsupported content type` is not actionable |
| Plain-text / Markdown / CSV ingestion | M5 ✅ | free: read and split, no parser |
| DOCX and ODT ingestion | M5 ✅ | ZIP + XML, ~30 lines each with the stdlib |
| Per-item attachment preference | M5 ✅ | ~100 items hold both a PDF and a snapshot; naive ingestion duplicates them |
| `page` generalised to a locator | M5 ✅ | HTML, Markdown, DOCX and notes have no pages |
| Linked-URL ingestion | **deferred**, §11.2.1 | makes the tool a web client; wants its own milestone |
| Wrap httpx transport errors | M5 ✅ | a Zotero timeout printed a traceback (§14.7) |
| Per-chunk embedding-model tag, `index --repair` | M4 | repair the stale chunks instead of rebuilding the store |
| Evaluation harness | M6 ✅ | §12.1 defines a golden set that nothing executes |
| Per-question source provenance in a golden set | M6 ✅ | one article per question is what makes hit rate discriminating (§12.2) |
| `generate-questions` | M6 ✅ §11.3.2 | measurement was bounded by how many questions a user will hand-write |
| `--merge` for incremental question sets | **deferred**, §11.3.2 | a second run replaces the file; adding only uncovered documents is the incremental form |
| `status` command | M9 | no way to inspect a folder's index |
| Stray-positional error message | M9 | `index Networking` fails opaquely |
| Cache size reporting and pruning | M9 | the cache only ever grows |
| Config schema version and migration | M11 | later settings must not break existing folders — and, per §11.4.1, an improved *default* never reaches a folder that pinned the old value |

**Not scheduled** — deferred because their value for *this* corpus is unproven,
not because they are uninteresting: agentic / multi-step RAG, hierarchical
summarisation retrieval, a web UI, cross-collection federated query, and
figure/table extraction from PDFs. Each should be reconsidered only with an M6
number showing the simpler path has run out.

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
a separate file. Since M6 they are also **executable**, as
`questions/Writing.yaml` — copy it into an index folder as `questions.yaml` and
run `zotero-rag eval`. That file must stay faithful to this section; a test parses
it on every run so a typo fails the suite rather than a live evaluation.

`expected_items` there lists the four `Writing` items that actually have an
indexed PDF, which encodes §12.7's criterion literally — "at least one correct
citation resolving to a real item in `Writing`". It deliberately does **not**
claim to know which of four overlapping how-to articles a given point came from,
because for this set that provenance is genuinely unknown.

### 12.2 The `Networking` golden set

Test collection: **`Networking`**, key **`SHWKWYDD`** — a *tree* of 8
collections, 18 items, 16 indexed documents (13 PDF + 3 HTML), 1,186 chunks. This
is the set that can actually measure a retrieval change, and the one §11.4 should
be run against.

**21 questions over 7 articles, three per article**, in `questions/Networking.yaml`,
generated from `questions/Networking.txt` (which remains the source of truth for
the wording of every question and reference answer). Nine are in Spanish and
twelve in English, over an English corpus, so the set exercises §5.3 both ways.

Two properties make it stronger than the `Writing` set:

- **One source article per question**, so `expected_items` names exactly the item a
  correct answer must rest on, and retrieval hit rate becomes a real discrimination
  test rather than a check that retrieval stayed inside the collection.
- **A verbatim prose reference answer *and* distilled expected points** per
  question. The prose is the more precise statement and is what the judge weighs;
  the points are the units coverage is counted in. Nothing in the points asserts
  anything the reference answer does not already say.

One trap worth recording, because it is the kind of thing that silently inflates a
score: the collection also holds `KJFG9YUE`, "Traffic engineering with
reinforcement learning in SDN", which reads like the CFR-RL paper but is a saved
Bing search page. It is deliberately not an expected item, so retrieving it is not
a hit.

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

Last updated: 2026-07-28. **470 tests passing; `ruff`, `ruff format` and `mypy` clean.**

### 14.1 Done

| Milestone | Scope | State |
|---|---|---|
| 0 | Package skeleton, config layer, `list-collections` | ✅ verified live |
| 1 | `init` (offline scaffolding) | ✅ verified live |
| 2a | Collection resolution + `index --dry-run` | ✅ verified live |
| 2b | Download, extract, chunk, embed, store, pin | ✅ verified live |
| 3 | `query`: retrieve, prompt, answer, cite | ✅ verified live |
| 4 | Store fingerprint, version-aware reindexing, repair, prune | ✅ verified live |
| 5 | HTML / text / DOCX / ODT / EPUB / notes, locators, preference | ✅ verified live |
| 6 | `eval`: golden sets, LLM judge, hit rate, coverage, comparable runs | ✅ verified live |
| 6b | `generate-questions`: a golden set written from the index itself (§11.3.2) | ✅ verified live |
| 7a | Windowed retrieval — built and measured; **rejected as a default** (§11.4.1) | ✅ verified live |
| 7b | Hybrid BM25 + vector retrieval — **accepted, now the default** (§11.4.2) | ✅ verified live |
| 7c | Cohere reranking — **accepted, recommended, off by default** (needs a key) (§11.4.3) | ✅ verified live |

**All eight acceptance criteria (§12) are verified against the real library**, as
are M4 (§14.6), M5 (§14.7), M6 (§14.8), §11.3.2 (§14.9), §11.4.1 (§14.10) and
§11.4.2 (§14.11). M7 continues with **retrieval diversity**, which §11.4.1 turned up
as the cheapest remaining win and which hybrid did not address, then metadata
filters and reranking — each accepted or rejected the same way.

**The scoreboard so far**, all on the 21-question `Networking` set:

| configuration | mean score | coverage | hit | cited | LLM tokens |
|---|---|---|---|---|---|
| PoC default — `top_k: 5`, vector | 1.81 | 53.3% | 86% | 76% | 33.3k |
| shipped default — `top_k: 10`, hybrid | 2.33–2.52 | 72.1–72.3% | 95% | 95% | 62.9k |
| recommended with a Cohere key — `top_k: 5`, hybrid, reranked | 2.38 | 73.8% | 95% | 95% | **33.9k** |
| best measured — `top_k: 10`, hybrid, reranked | **2.62** | **75.5%** | 95% | 95% | 63.4k |

Read against the PoC baseline, the share of answers citing the right paper went 76% →
95% and point coverage 53% → 76%. The instructive part is *how* each lever paid:
`top_k` bought quality with tokens, hybrid bought it for free, reranking bought it
while *giving tokens back* — and windowing was declined outright by the same
measurement. None of those four conclusions was predictable from the design, which is
the whole return on M6.

Folders created before 0.4.4 keep `top_k: 5` until edited; see the end of §11.4.1.

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
  cli.py           typer app: list-collections, init, index, query, eval,
                   generate-questions
  config.py        pydantic models, layered loading, pin_collection()
  scaffold.py      init: config.yaml + .env + .gitignore
  zotero_api.py    client, collection tree, path/key resolution, item fetch
  planning.py      build_plan(): §7.1 selection, preference, notes, chunk_metadata()
  retrieval.py     BM25 keyword index, reciprocal rank fusion, retrieve() (M7)
  reranking.py     Cohere cross-encoder reranking of candidates (M7, §11.4.3)
  evaluation.py    golden-set loading, LLM judge, metrics, EvalReport (M6)
  generation.py    generate-questions: reassemble documents, write a set (§11.3.2)
  documents.py     formats, content-type mapping, locator kinds
  html_text.py     stdlib HTML → sections (snapshots and notes)
  office.py        DOCX / ODT / EPUB via zipfile + xml.etree
  metadata.py      authors_str(), parse_year(), zotero_url()
  download.py      cache + multipart-envelope repair
  extraction.py    segment extraction, dispatched by format
  chunking.py      RecursiveCharacterTextSplitter per segment
  embedding.py     OpenRouter embeddings + tiktoken counting
  vectorstore.py   chromadb wrapper, StoredChunk
  indexer.py       orchestration, IndexStats, failure isolation, repair, prune
  fingerprint.py   StoreFingerprint, intent-vs-fact comparison, severities
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

cp <repo>/questions/Writing.yaml questions.yaml    # M6, §11.3
uv run zotero-rag eval
```

Expected: 4 PDFs, 22 pages, 146 chunks, ~30,900 embedding tokens, ~7s. A second
`index` reports 0 indexed / 4 skipped. The query answers in Spanish in ~3s,
citing 2–3 real `Writing` items with working `zotero.org` URLs.

**The figures recorded in §12 and §14.2 predate 0.4.3/0.4.4** and were measured at
`top_k: 5` with vector retrieval — §12.7's "5 chunks, 1,287 prompt + 181 completion
tokens" is that configuration, not this one. A run today retrieves 10 chunks and fuses
BM25 with the vectors, so expect roughly double the prompt tokens and possibly a
different set of cited pages. The acceptance criteria themselves still hold; only the
counts moved. Add `--top-k 5 --retrieval vector` to reproduce the original numbers
exactly.

Expected from `eval`: 3 questions, retrieval hit 100%, expected item cited 100%,
mean score ~1.3/5 and point coverage 22–40% — the last two are *low on purpose*,
and §14.8 explains why that is the correct reading of this collection rather than a
misconfiguration. For a set large enough to compare settings, use
`questions/Networking.yaml` against `Networking` (§12.2).

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
  reranker or a larger `top_k` is the §11.4 lever if that matters.
- **Known defect, unguarded in the PoC: the store does not record which embedding
  model wrote it.** Verified behaviour — swapping `embedding_model` for another of
  the same dimension yields meaningless neighbours with no error, while a
  different dimension raises `Collection expecting embedding with dimension of N`.
  Editing `embedding_model` in a populated index folder is therefore unsafe today;
  delete `chroma/` and re-index instead. Fixed by the fingerprint in §11.1, which
  is why M4 leads the roadmap.

### 14.6 M4 as built and verified

Verified against the real `Writing` store (146 chunks) and the pre-M4 `Networking`
store (1,150 chunks), from a global `uv tool install`:

| Check | Result |
|---|---|
| Adoption of a pre-M4 store | fingerprint written, dimension 1536 derived from the data, one warning printed, nothing re-embedded |
| Same-dimension model swap (`3-small` → `ada-002`) | `query` **refused**, naming both models and offering `--repair` / `--force`; the previously silent corruption is now impossible |
| `index --repair` | re-embedded all 146 chunks (30,863 tokens, no downloads), store then reported one model and zero stale chunks, `query` worked |
| `index --force` | discarded 146 chunks, re-embedded to exactly 146 chunks from 22 pages, cache untouched (4 files, zero downloads) |
| `index --force --refetch` | cache cleared, all 4 PDFs downloaded again, store rebuilt |
| Orphan reporting | a ghost attachment was reported by a normal run and *kept* |
| `index --prune` | removed exactly that attachment's chunk, leaving 146 |
| Regression | 236 tests, `ruff`, `ruff format`, `mypy` all clean |

`write_fingerprint` was explicitly proved harmless to retrieval: with a fixed query
vector against two copies of the same store, the top-5 ids and distances are
bit-identical before and after the fingerprint is written. Worth stating because
`collection.modify()` touching a collection's metadata is exactly the kind of call
that could quietly rebuild an index.

### 14.7 M5 as built and verified

Verified live against the real `Networking` collection (8 collections, 18 items)
and the `Writing` regression, from a global `uv tool install`:

| Check | Result |
|---|---|
| Coverage on `Networking` | **16 documents from 16 items** (13 pdf + 3 html), up from 13 PDFs |
| The three previously invisible items | exactly the ones the pre-M5 run warned about as having no PDF; now indexed as 36 chunks from 29 sections (2,912 tokens) |
| Duplicate suppression | 7 snapshots recorded as `another attachment of this item is preferred`, all `text/html` — no double-embedding |
| `no readable attachment` | fell from 5 items to 2 |
| Skip reporting | each reason now lists its content types underneath |
| Section citations | a snapshot-only item answers with `§ Abstract:` and a working Zotero URL |
| HTML extraction end to end | a hand-built snapshot indexed to 3 sections labelled `Introduction`, `Why fog is needed`, `Fog versus edge computing`; `nav`, `script` and `footer` text absent; query answered with `§ Fog versus edge computing` |
| PDF regression | `Writing` unchanged at 146 chunks, citations still `p. N`, summary still says `pages` |
| Notes | indexed with no download at all (a test fails the run if a download is attempted) |
| Regression | 295 tests, `ruff`, `ruff format`, `mypy` all clean |

**Two things found by running it, not by planning it:**

- **A Zotero timeout printed a full traceback.** pyzotero speaks HTTP through
  `httpx`, so a `ConnectTimeout` is not a `PyZoteroError` and escaped every
  `except` in `zotero_api.py`. The Zotero API was genuinely flaky during the M5
  run, which is how this surfaced. Network failure is an *expected* condition for
  this tool (§10), so `NETWORK_ERRORS` now covers `PyZoteroError`, `httpx.HTTPError`
  and `OSError`, and `httpx` is declared as a direct dependency rather than used
  as a transitive one. The same failure then read:
  `Error: Could not fetch collections from Zotero: The read operation timed out`.
- **The summary called snapshots "PDFs".** `Indexed 3 PDFs (36 chunks from 29
  pages)` was wrong in both nouns. The wording now follows the formats actually
  indexed: `documents`, and `pages` / `sections` / `pages and sections`.

**A caveat about what snapshots contain.** The three newly indexed snapshots are
publisher landing pages (IEEE Xplore and similar), so their sections are
`Abstract:` and the article title — not full text. Coverage genuinely improved,
and those items are now answerable, but a snapshot of a paywalled article is an
abstract, not the paper. This is the §11.2 prediction confirmed: a snapshot of an
article that also has a PDF is usually the abstract page plus furniture, which is
exactly why the PDF wins by default.

### 14.8 M6 as built and verified

Verified live against both collections, freshly indexed: `Writing` reproduced its
recorded regression exactly (4 documents, 146 chunks from 22 pages, 30,863
embedding tokens) and `Networking` reproduced M5's (16 documents, 1,186 chunks from
211 pages and sections, 271,417 tokens).

| Check | Result |
|---|---|
| `eval` on `Writing` (3 questions) | ran, graded and reported; **retrieval hit 100%, expected item cited 100%** — the §12.7 citation criterion reproduced |
| `eval` on `Writing`, point coverage | **22–40% across runs, one question `wrong`** — §12.7's "covering most of the expected points" does *not* survive measurement (see below) |
| `eval` on `Networking` (21 questions) | mean 1.81/5, hit 86%, cited 76%, coverage 53–56%, 2 good / 15 partial / 2 wrong / 2 no answer, ~58k tokens in ~100s |
| `top_k: 1` scores measurably worse | **yes on `Networking`** — 0.76 vs 1.81 mean, 29% vs 76% citation rate. **No on `Writing`**, where it scored *better*: a 3-question set cannot resolve the difference (§11.3.1) |
| Repeat run, identical settings | grade labels, hit rate and citation rate **bit-identical**; point coverage moved 53.3% → 56.0%. Temperature 0 is necessary, not sufficient |
| `--json` | full report — `settings`, `aggregate`, `results[]` — on stdout, progress on stderr, so the payload is never contaminated |
| `--only`, `--top-k`, `--assess-context` | all exercised; an unknown `--only` id lists the available ones |
| Golden sets parse in CI | both shipped sets are loaded by the test suite, so a typo fails `pytest` rather than a paid run |
| No new dependency | `pyyaml` and `langchain-openai` were already required; §9 holds |
| Regression | 348 tests, `ruff`, `ruff format`, `mypy` all clean |

**The `Writing` coverage result is a real finding, not a harness defect.** Reading
the answers confirms the grades. The `publicacion` question — "¿Qué pasos involucra
el proceso de publicación?" — retrieved three chunks *all from Driscoll & Aquilina*
and answered with that paper's six-step **writing** process: read to write, select
the journal, plan, contact the editorial team, use the journal as a template,
prepare the manuscript. The expected points are about what happens *after*
submission: peer-review revisions, the editor's acceptance letter, the galley
proof. The answer is on-topic, fluent, correctly cited, and does not answer the
question — which is precisely the failure mode an eyeball review passes and a
golden set catches. The fix belongs to M7 (this looks like a retrieval-diversity
problem), not to the judge.

**What the `Networking` numbers point M7 at.** Of 21 questions, 3 never retrieved
their source article at all (`jellyfish-3`, `wang-2`, `wang-3`) and 2 more
retrieved it but never cited it (`cfr-rl-3`, `pathan-1`). That split is the whole
argument for keeping a mechanical hit rate beside an LLM grade: the first three are
retrieval failures and the second two are generation failures, and no single
quality score would have told them apart.

**Judge cost is not negligible and is reported separately.** The `Networking` run
spent 31,097 prompt + 2,172 completion tokens answering and a further 24,234
grading — grading is ~42% of the bill. Conflating the two under one "tokens"
figure would have hidden that, which is why `judge_tokens` is its own field.

### 14.9 `generate-questions` as built and verified (§11.3.2)

Verified live against the indexed `Networking` store (16 documents, 1,186 chunks):

| Check | Result |
|---|---|
| `--dry-run` | listed all 16 documents with chunk and character counts, estimated **97,780 prompt tokens**, wrote nothing, made no model call |
| Estimate accuracy | the real run spent **106,222** prompt tokens — the dry run was within 8%, so it is usable for deciding whether to proceed |
| Generation | **48 questions from 16 documents** (3 each), 106,222 prompt + 7,507 completion tokens, 129s |
| Head-and-tail sampling | 12 documents exceeded the 30,000-character budget and were sampled, reported in the summary |
| `eval` on the generated set, **unedited** | ran all 48: mean **2.08/5**, retrieval hit **81%**, expected item cited **73%**, coverage **61.6%**, 4 perfect / 5 good / 30 partial / 3 wrong / 6 no answer |
| Comparable to the hand-written set | the 21-question hand-written set scored 1.81 / 86% / 76% / 53% — close enough on every metric that neither set looks anomalous, which is the mutual validation worth having |
| Overwrite refusal | an existing `questions.yaml` is refused with `--force` named, **before** any generation call; the file is left byte-identical |
| Failure isolation | a document whose generation fails is named in the summary and the run still writes the rest |
| Round-trip guarantee | the written file is re-parsed by `load_questions` before the command returns |
| Regression | 389 tests, `ruff`, `ruff format`, `mypy` all clean |

**Question quality is better than expected, and independently checkable.** The
generated `zhang-2020-1` — "What problem does CFR-RL address in traffic engineering
and how does it propose to solve it?" — is very nearly the hand-written `cfr-rl-1`
of §12.2, arrived at independently from the same paper. That correspondence is the
strongest evidence available that the generator is asking about main points rather
than trivia.

**Two ugly ids, found by running it and fixed.** Both were in the derived question
ids, which matter because `--only` takes them:

- `Fröhlich` became **`fr-hlich`**: the slug regex stripped the `ö` as a
  non-ASCII character and left a gap. Now the name is NFKD-decomposed and its
  combining marks dropped, so it slugs to `frohlich`.
- A title-derived stem was cut mid-word at 32 characters —
  **`traffic-engineering-with-reinfor`**. Now trimmed back to the last word
  boundary, giving `traffic-engineering-with`.

**The generator will faithfully write questions about a worthless document.** All
three questions generated from `KJFG9YUE` — the saved Bing search page of §12.2 —
were retrieval misses in the evaluation, because a search-results page has nothing
the rest of the corpus can answer about. Nothing malfunctioned: the command
generated from what was indexed, and evaluation then correctly reported those
questions as unanswered. It is a clean demonstration of why the output is a draft
to review, and `--only-item` is the tool for excluding such documents up front. It
is also an argument for the M9 `status` command: junk in a collection is easier to
notice when something lists what was indexed.

### 14.10 Windowed retrieval as built and measured (§11.4.1)

Verified live against the indexed `Networking` store (16 documents, 1,186 chunks),
nine `eval` runs over the 21-question set of §12.2.

| Check | Result |
|---|---|
| Mechanism | `--top-k 2 --window 1` expanded 2 hits to 4 chunks and answered correctly with CFR-RL's `r = 1/U` reward, citing p. 3 |
| `window: 0` is byte-identical to the pre-M7 path | the hits are returned untouched; no store call is made at all |
| Dedup | two adjacent hits with overlapping windows emit each chunk once |
| Page-boundary crossing | a hit on the first chunk of page 2 pulls the last chunk of page 1 — the one join `chunk_overlap` does not cover |
| Distances survive | the refetched copy of a hit is replaced by the retrieved object, so `Source.distance` is still populated |
| Recorded in the report | `retrieval_window` appears in `settings`, and the summary line reads `vector ±2 retrieval` |
| **Quality verdict** | **rejected as a default** — every windowed run is dominated by or tied with a cheaper non-windowed one (§11.4.1) |
| Regression | 406 tests, `ruff`, `ruff format`, `mypy` all clean |

**What made the verdict trustworthy rather than a coin flip.** The comparison that
mattered was not "does windowing help?" — it does, at `window: 2` — but "does it
help more than spending the same tokens on `top_k`?" Answering that needed the
non-windowed configurations at matched cost, so `top_k` 20 and 30 were measured too.
Without those two runs the honest conclusion would have been "windowing improves
quality at 3× the cost", which is true and tells nobody what to do. §11.3.1's noise
floor did the rest of the work: it is what licenses calling a 1.8-point coverage gap
a tie instead of a result.

**A bug this work fixed in passing.** `VectorStore.get_by_attachment()` sorted its
chunks by *id string*, so page 10 came back before page 2 and an attachment was
handed to the caller scrambled. Nothing depended on the order before — its callers
wanted a single chunk or built a dict — but windowed retrieval does, and so did
`generate-questions`, which had quietly needed its own numeric sort. Both now share
`reading_order_key()` in `vectorstore.py` and the duplicate is gone.

**The cost of a rejected feature was ~9 eval runs and one afternoon**, which is the
argument for M6 restated: the alternative was shipping `retrieval_window: 2` as a
default because the mechanism is elegant and one answer looked better.

### 14.11 Hybrid retrieval as built and measured (§11.4.2)

Verified live against the indexed `Networking` store (16 documents, 1,186 chunks),
five `eval` runs across two question sets.

| Check | Result |
|---|---|
| Fusion engages | `-v` reports `hybrid: 20 vector + 20 keyword candidate(s) fused to 3` |
| Keyword index built once per process | monkeypatched counter proves one build for a multi-question `eval`; `vector` mode builds none at all |
| Build cost | **0.21s for 1,186 chunks** (0.10s reading Chroma, 0.11s tokenizing + BM25) |
| Acronym query end to end | `--retrieval hybrid` answered "¿Qué es rEDKSP?" citing Jellyfish pp. 6 and 9 — as did vector, so this one was never the differentiator |
| **Quality verdict** | **accepted** — better on both sets at nil token cost; citation rate 81% → 95% at `top_k: 10` (§11.4.2) |
| Default changed | `retrieval: hybrid` shipped; a test pins the packaged default so it cannot drift back silently |
| No degenerate retrievals | across all hybrid runs, no row retrieved zero chunks and none under-filled `top_k` |
| Regression | 442 tests, `ruff`, `ruff format`, `mypy` all clean |

**A bug the tests caught before any user could.** The first implementation filtered
BM25 results on `score > 0`, on the reasoning that a zero score means the chunk holds
no query term. That reasoning is wrong. `rank_bm25` uses the Robertson–Sparck-Jones
IDF, `log(N − df + 0.5) − log(df + 0.5)`, which is **exactly zero when a term appears
in half the documents** and negative above that. So a genuine match can score zero —
reliably on a small store, where `df = N/2` is easy to hit — and a two-chunk test
corpus made `search("fog")` return nothing at all. The filter now tests token
presence directly, with the token sets kept per chunk for the purpose. Worth
recording because the wrong version would have looked perfectly healthy on the
1,186-chunk store and quietly failed for anyone with a small collection.

**Why two question sets earned their keep.** Hybrid moved retrieval hit rate by nine
points on the generated set and not at all on the hand-written one. Neither number is
wrong: generated questions echo the vocabulary of the paper they were written from,
which is what BM25 matches, while hand-written questions are paraphrases, which is
where dense retrieval already wins and BM25 has nothing to add. One set alone would
have supported either "hybrid transforms retrieval" or "hybrid does nothing for
retrieval", and both would have been overclaims.

**Windowing and hybrid, side by side**, since they were assessed the same way and
came out opposite:

| | windowing (`window: 2`) | hybrid |
|---|---|---|
| Token cost | 2–3× | ~1.0× |
| Retrieval hit rate | cannot change it, by construction | +9 points where terms are distinctive |
| Citation rate | −5 points at `top_k: 5` | +14 points at `top_k: 10` |
| Beaten by spending the same tokens on `top_k`? | yes | there are no tokens to respend |
| Outcome | kept, off by default | **default** |

The asymmetry is the lesson: windowing elaborates documents already found, hybrid
changes *which* documents are found. Only the second can move the metric that was
actually binding.

### 14.12 Reranking as built and measured (§11.4.3)

Verified live against the indexed `Networking` store, two `eval` runs plus a
single-query trace, using Cohere `rerank-v4.0-fast`.

| Check | Result |
|---|---|
| The model exists and behaves | probed directly before building anything: given three passages and a CFR-RL question it scored the right one 0.908, the Jellyfish one 0.165, and dropped the mercury one entirely |
| Composes with hybrid | `-v` shows `hybrid: 80 vector + 80 keyword candidate(s) fused to 20` then `reranked 20 candidate(s) down to 5` |
| Composes with windowing, in the right order | reranking picks the hits, then windowing expands them; a test asserts the neighbour reaches the model without having competed as a candidate |
| Fanout honoured | `candidates_considered` is 20 at `top_k: 5` and 40 at `top_k: 10` |
| Applied to every question | 21/21 in both runs — no silent fallbacks |
| **Quality verdict** | **accepted** — best result recorded, and the first lever to improve quality while *reducing* context (§11.4.3) |
| Missing credential | `--reranker cohere` without `COHERE_API_KEY` fails naming the variable, before any question is embedded |
| Misconfiguration vs. outage | a 4xx raises `RerankError` and stops; a 5xx or timeout warns, keeps the retrieved order, and is reported as not reranked |
| Off by default | pinned by a test, so it cannot start demanding a credential by accident |
| No new dependency | `httpx` was already required; §9 holds, and a local cross-encoder would have breached it |
| Regression | 470 tests, `ruff`, `ruff format`, `mypy` all clean |

**The result worth remembering**, because it inverts the pattern of the two levers
before it: `top_k: 5` with reranking scored 2.38 with 73.8% coverage on **33.9k** LLM
tokens, against `top_k: 10` without it at 2.38 and 67.9–72.3% on **61.5–62.9k**. Equal
or better quality on half the context. Windowing bought quality at 2–3× context and
was rejected; hybrid bought it at 1×; reranking bought it at 0.5×.

**A caveat recorded deliberately, because the table invites the wrong reading.** Those
are LLM tokens. Reranking makes one Cohere request per question — 20 to 40 documents —
billed in search units by a provider our accounting does not touch at all (verified:
the run's token buckets are prompt, completion, embedding and judge; there is no
rerank bucket). "Half the tokens" is therefore a statement about *context*, not about
*money*, and at these volumes the LLM saving is small in absolute terms while the
rerank charge is not obviously smaller. Resolving that is M8's pricing table, and
until then the claim stands only as quality-per-context-token. Latency is measurable
today: about one extra second per query.

**Why the switch is off despite winning.** It needs `COHERE_API_KEY`. Turning it on by
default would make `query` fail outright for anyone holding only the credentials §5.2
asks for — exchanging a tool that works for one they cannot run. Hybrid could become a
default because it cost nothing and needed nothing; reranking cannot, on the second
count alone. The recommendation therefore lives in the documentation instead, which is
the right place for advice that depends on what a user has.

**Three levers have now failed on `jellyfish-3`** — `top_k` to 30, keyword fusion, and
reranking over 40 candidates. A question that names nothing from its own paper cannot
be retrieved, however much retrieval is thrown at it. That is now the best-evidenced
finding in §11.4, and it points at metadata filters rather than at anything ranking can
do.
