# Design of a terminal UI for zotero-rag

A full-screen terminal UI for asking questions of an index folder, limited to a chosen
subtree of the pinned collection. It sits next to the `zotero-rag` CLI (PRD.md §3) and
does not replace it. It uses the same config, store and query pipeline (PRD.md §8).

## 1. Goal

Let a user stay in one screen, choose which part of the pinned collection to search,
ask a series of questions, and read cited answers as they stream in, without running
`zotero-rag query` again for every question.

### 1.1 Non-goals for v1

- Changing retrieval settings during a session (top_k, window, retrieval, reranker).
  These are fixed at launch (§5).
- Follow-up questions. Every question is answered on its own, and earlier turns are
  not sent to the LLM (PRD.md §11.6 `chat` remains a separate item).
- Indexing, `--repair`, or any other change to the store from inside the UI.
- Browsing collections outside the pinned subtree.

### 1.2 Deferred to v2

- Cancelling an in-flight query without quitting.
- Exporting the session transcript as Markdown.
- Copying an answer to the clipboard.

### 1.3 Future consideration

- Indexed document or chunk counts next to each folder in the tree.

## 2. Toolkit and entry point

- **Toolkit:** [Textual](https://textual.textualize.io/). It comes from the same team
  as `rich`, which is already a dependency. It provides `Tree`, `Markdown`, `TextArea`,
  async workers, mouse support and resize handling. Add `textual` to the
  `pyproject.toml` dependencies.
- **Entry point:** a separate console script, `zotero-rag-ui`, declared in
  `[project.scripts]` next to `zotero-rag`. It lives in a new module
  `src/zotero_rag/tui/` (app, widgets, CSS).
- **Working directory:** like `query`, it must be run from an index folder. It uses
  the same `_open_index()` checks: config present, store present and non-empty,
  embedding fingerprint matching (PRD.md §11.1). If a check fails, it prints the
  same error the CLI prints, exits with the same non-zero code, and never draws the
  screen. Move `_open_index`, `_resolve_reranker` and `_build_keyword_index` out of
  `cli.py` into a module both entry points can import.

### 2.1 Launch flags

```
zotero-rag-ui [--top-k N] [--window N] [--retrieval vector|hybrid]
              [--reranker none|cohere] [-v/--verbose] [--version]
```

These mean the same as they do for `query` and override `config.yaml` in the same
way. `-v` sends diagnostic logging to a log file (§8), never to the screen.

## 3. Layout

```
┌──────────────── Tree ───────────┬──────────────── Results ────────────────────────┐
│ ▾ Networking            (root)  │ ── [Networking/Routing] ─────────────────────── │
│   ▸ Routing                     │ > What is BGP route flap damping?               │
│   ▾ Transport                   │                                                 │
│       TCP                       │ Route flap damping suppresses … [1][3]          │
│       QUIC                      │                                                 │
│   ▸ Security                    │ Sources                                         │
│                                 │  [1] Villamizar et al. (1998) RFC 2439, p. 4    │
│                                 │      https://www.zotero.org/…                   │
│                                 │  [3] …                                          │
│                                 ├──────────────── Prompt ─────────────────────────┤
│                                 │ How does QUIC handle head-of-line blocking?▌    │
│                                 │                                                 │
├─────────────────────────────────┴─────────────────────────────────────────────────┤
│ Scope: Networking/Transport · hybrid · top_k 10 · rerank cohere · gpt-4o-mini      │
│ Tab focus  Enter send  Alt+Enter newline  ↑↓ history  ? help  Ctrl+Q quit         │
└───────────────────────────────────────────────────────────────────────────────────┘
```

- **Left, 1/3 of the width:** the folder tree (§4).
- **Right, 2/3 of the width:**
  - **Top, 3/4 of the height:** the scrollable results transcript (§6).
  - **Bottom, 1/4 of the height:** the multi-line prompt editor (§7).
- **Status bar:** active scope path, retrieval mode, top_k, window (when it is not 0),
  reranker, and LLM model. While a query runs it also shows a spinner and the phase
  (retrieving, answering).
- **Footer:** the key hints for the focused pane.
- The proportions follow terminal resizes. Under 80×24, show a "terminal too small"
  message rather than a broken layout.

## 4. Folder tree

### 4.1 Source: a snapshot written at index time

The UI does not call the Zotero API. The tree comes from a snapshot of the pinned
subtree that `index` saves, so the UI:

- starts instantly and works offline;
- only shows folders that the store's metadata knows about (§4.3).

**Change to `index`** (✅ built): after every non-dry run, including `--force`,
`--prune` and `--repair`, write `<index-folder>/collections.json`
(`snapshot.py`). It holds a format `version`, a `written_at` timestamp and `roots`:
a list of `CollectionNode` trees (key, name, children). For a pinned collection that
is the single pinned subtree, without children when `recursive: false`. For a
whole-library index it is every library root. The file is written atomically and is
listed in PRD.md §6. The `.gitignore` that `init` creates does not ignore it.

If the snapshot is missing, because the store was indexed by an older version, the
UI starts with only the pinned root node and tells the user once in the transcript:
"Folder filtering needs `zotero-rag index` to be run once more." (see §4.3.2).

### 4.2 Behaviour

- On launch the root (the pinned collection) is selected and expanded one level.
  Searching from the root means searching the whole store, with no filter.
- Selecting a node (Enter, Space or a mouse click) makes it the **active scope**. A
  search scoped to a node covers that node **and all its descendants**.
- Arrow keys move, and ←/→ collapse and expand. Moving the cursor does not change the
  scope. Only selecting does.
- The active scope node stays highlighted, separately from the cursor.
- Changing the scope only affects questions asked afterwards.

### 4.3 Scope filtering: collection keys in chunk metadata

Chunks today carry `collection_name`, the name of a single collection (PRD.md §7.2).
That cannot filter by subtree. Add:

| field | source |
|---|---|
| `collection_keys` | the keys of every collection in the pinned subtree that contains the item. Ancestors are not stored. |

An item can sit in several collections, so the field holds several keys.

At query time, the selected node is expanded into the set of keys for its whole
subtree, taken from the snapshot. A chunk is in scope if any of its `collection_keys`
is in that set. Doing the expansion at query time keeps the metadata stable when the
user moves folders around in Zotero; only membership has to be refreshed.

#### 4.3.1 Applying the filter (✅ built)

A scope is a collection of keys (`vectorstore.Scope`). `None` means no filter. The
UI gets one from `CollectionSnapshot.subtree_keys(key)`, which returns the selected
folder's key and every descendant's key.

- **Vector search:** `VectorStore.query(embedding, top_k, scope)` passes the filter
  built by `scope_where`. Without a scope it makes exactly the call it made before.
- **BM25 (hybrid):** `KeywordIndex.search(question, limit, scope)` applies
  `in_scope`, the same rule in Python, **before** ranking, so a narrow folder still
  fills its limit. The BM25 index is still built once per session over the whole
  store, so IDF reflects the whole library.
- **Empty scope:** both searches return nothing without calling Chroma. A chunk with
  no `collection_keys` is outside every scope but inside an unscoped search.
- **Reranking and windowing:** these run on the candidates that are already in scope.
  Window neighbours come from the same attachment, so they are in scope too.
- `answer_question` (and `retrieve`) take an optional `scope`, defaulting to None
  (no filter). `query` and `eval` keep passing None, so their results do not
  change.

**Encoding (verified against chromadb 1.5.9, 2026-10-28):** `collection_keys` is
stored as a native **list of strings**. Results from a scratch store:

| filter on `collection_keys` | result |
|---|---|
| `{"$contains": "K"}` | ✅ matches chunks whose list contains `K`, in both `get` and `query` |
| `{"$or": [{"$contains": "K1"}, {"$contains": "K3"}, …]}` | ✅ union. A 201-clause `$or` works. |
| `{"$not_contains": "K"}` | ✅ also matches chunks with **no** `collection_keys` field |
| `{"$in": ["K1", "K3"]}` | ⚠️ **silently returns nothing.** `$in` does not apply to lists. Never use it. |
| `"K"` (implicit `$eq`) | ⚠️ silently returns nothing |
| `$or` with a single clause | ❌ `ValueError`: `$or`/`$and` need at least two expressions |
| empty list `[]` on `add` | ❌ `ValueError`: list values must be non-empty |

Rules for the implementation:

- Build the filter from the expanded key set like this: one key → a bare
  `{"collection_keys": {"$contains": k}}`; two or more → `$or` of `$contains`
  clauses. Unit-test both shapes, because the wrong operator fails silently.
- An item must always have at least one key. Every indexed attachment comes from some
  collection in the pinned subtree, so this holds. Assert it at write time rather
  than writing `[]`.
- List values survive reopening the persistent store, and `update(metadatas=…)`
  replaces the list in place, which is what the membership backfill needs.
- Subtrees in a personal library are well under the 200 keys tested. If one is ever
  larger, retest `$or` at that size before falling back to filtering in Python.

Record this encoding in PRD.md §7.2.

#### 4.3.2 Migrating existing stores: any `index` run (✅ built)

This needs no new flag. `index` already fetches every collection in scope, so it knows
each item's membership exactly (`planning.item_membership`). Every run, including
`--repair`, which runs the normal pass after re-embedding:

1. writes `collection_keys` on the chunks of new and changed attachments as they are
   embedded;
2. compares every planned attachment's keys with what is stored and rewrites those
   that differ (`indexer.refresh_collection_keys`). This is a metadata-only Chroma
   `update` with no re-embedding and no downloads. It catches both stores written
   before the field existed and items moved between folders, which do not bump the
   attachment version. The summary reports "Updated the folder membership of N
   attachments (no re-embedding)";
3. then writes `collections.json`.

**Store-readiness detection:** the snapshot is written only after step 2 has
finished, so **its presence is the readiness flag**. There is no separate store flag
and no sampling of chunks. Chunks of orphaned attachments that are still in the store
(not `--prune`d) keep their old keys or have none. That is harmless: they are not in
the snapshot's scope, so a scoped search can't reach them, and an unscoped search
from the root does not filter. When filtering is unavailable:

- the tree shows only the root and can't be expanded;
- questions still work against the whole store;
- the transcript shows the "run `zotero-rag index`" notice once.

## 5. Settings

Settings come only from `config.yaml` and the launch flags (§2.1). They are read once
at startup and shown in the status bar. The keyword index, embeddings client, chat
client and reranker are each built once per session.

## 6. Results transcript

- A **scrolling transcript** of the session. Each question and its answer are
  appended at the bottom, and earlier entries stay available for scrolling.
- Each entry has:
  1. a **scope tag** header with the scope path the question was asked under, e.g.
     `[Networking/Transport]`;
  2. the **question**, as typed;
  3. the **answer**, rendered as Markdown and **streamed** as tokens arrive (§6.1);
  4. a **Sources** list of the **cited** sources only, as `[n] label` plus the
     Zotero URL. This matches the CLI behaviour in PRD.md §8.2, including the
     "The answer cites no sources; N passages were retrieved" line when nothing is
     cited.
- New content auto-scrolls to the bottom unless the user has scrolled up, in which
  case it stays where they are and a "new output ↓" hint appears.
- When the results pane has focus: PgUp/PgDn/Home/End and arrow keys scroll, and the
  mouse wheel scrolls everywhere.
- Errors (`ZoteroRagError` from retrieval, `LLMError`, a reranker failure) are shown
  inline as an error entry under the question. The session keeps going.
- A query with no in-scope chunks gives the usual `NO_CONTEXT_ANSWER`, without
  calling the LLM, and adds a note that the scope may be too narrow.

### 6.1 Streaming

`answer_question` returns a finished `QueryResult`. Streaming needs a variant, in
`querying.py` so the CLI can adopt it later (PRD.md §11.6), with this shape:

- run retrieve → rerank → window → build context exactly as now;
- stream the LLM response (`ChatOpenAI.stream`) and send text deltas to an
  `on_token` callback;
- when the stream ends, build the same `QueryResult`: parse citations, mark `cited`,
  and take usage from the last chunk's `usage_metadata`. That needs `stream_usage`
  enabled on the client.

Refactor `answer_question` into shared pieces so the two variants cannot drift.

**Built:** `stream_answer_question(..., stream, on_token, ...)` in `querying.py`.

- Both variants run the same `_prepare` (retrieve → rerank → window → prompt) and
  `_finish` (citations, usage, counts) steps. A test checks that, given the same model
  text, they return identical results.
- `stream` takes the messages and yields `AIMessageChunk`s (`chat.stream`). The
  chunks are summed into one message, which merges the text and the usage sent on the
  final chunk. `build_chat` sets `stream_usage=True`.
- A provider error at any point in the stream, including mid-answer, is raised as
  `LLMError`. Deltas sent before the error have already reached `on_token`. An
  exception raised by `on_token` itself propagates unchanged.
- With nothing retrieved, the model is not called and `NO_CONTEXT_ANSWER` is sent to
  `on_token` as a single delta.

The UI renders the Markdown progressively. To avoid flicker, re-render no more than
about 10 times per second. The Sources list appears only after the stream completes,
because citations are only known then.

The query runs in a Textual worker thread, so the UI stays responsive. While it runs,
sending another question is disabled (the prompt stays editable). Cancelling is v2
(§1.2).

## 7. Prompt editor

- A **multi-line** `TextArea`. **Enter** sends, and **Alt+Enter** (also
  **Shift+Enter** where the terminal reports it) inserts a newline.
- An empty or whitespace-only prompt is not sent.
- **History:** ↑ on the first line or ↓ on the last line steps through earlier prompts.
  A recalled prompt can be edited before sending, and sending it adds it as a new
  entry. Whatever was being typed is kept as a draft and comes back when the user
  steps past the newest entry.
- **Persistence:** history is stored per index folder in
  `<index-folder>/.zotero-rag-ui/history`, one JSON-encoded string per line so
  multi-line prompts survive. It is loaded at startup and appended after each send.
  Keep the last 1000 entries and skip consecutive duplicates. Add the directory to the
  `.gitignore` that `init` writes, and to PRD.md §6.

## 8. Navigation and keys

| key | action |
|---|---|
| Tab / Shift+Tab | cycle focus: prompt → tree → results |
| Enter (prompt) | send |
| Alt+Enter / Shift+Enter | newline in the prompt |
| ↑ / ↓ (prompt, at first/last line) | history |
| ↑ ↓ ← → / Enter / Space (tree) | move / collapse / expand / select scope |
| PgUp PgDn Home End (results) | scroll |
| F1 (anywhere) / ? (outside the prompt) | help overlay listing every binding |
| Ctrl+Q | quit |

- Focus starts on the **prompt**.
- The mouse works throughout: clicking focuses a pane, selects a tree node, or scrolls.
- Ctrl+C does not quit in v1. It is reserved for cancelling in v2. Ctrl+Q is the only
  way to quit.
- `-v` writes logs to `<index-folder>/.zotero-rag-ui/ui.log`, because stderr can't be
  used while the screen is drawn. Noisy third-party loggers are quietened as
  `_set_noisy_logger_level` does.

## 9. Non-functional requirements

- **Startup:** the first screen appears in under 1 s, not counting BM25 index
  construction. If hybrid mode is on, the keyword index builds in a worker and the
  status bar shows "building keyword index…". Questions sent before it finishes wait
  in a queue.
- **No network at startup.** Network is used only by queries (embedding, LLM,
  reranker).
- **Secrets:** API keys never appear on screen or in the log.
- **Terminal:** works in standard xterm-compatible terminals on Linux and macOS, with
  a minimum size of 80×24. Windows is best-effort through Textual.
- **Quality bar:** the same rules as the rest of the tree (ruff, mypy strict on
  `src/` and `tests/`).

## 10. Testing

- **Unit:** scope expansion from a snapshot; the `collection_keys` filter in both the
  vector and BM25 paths; the membership backfill done by `index` (with the Zotero client faked as
  in the existing tests); `collections.json` round-trip; history load, append, trim
  and draft behaviour; the streaming variant gives the same `QueryResult` as the
  non-streaming one for the same fake LLM output.
- **UI:** Textual's `App.run_test()` / `Pilot` with a fake store and fake LLM:
  launch, select a scope, send a question, check that the streamed answer and the
  cited sources appear in the transcript under the right scope tag, recall history
  with ↑, and the unavailable-filtering path.
- **Regression:** `query` and `eval` results stay the same when no scope is given.

## 11. Implementation order

1. ✅ Move the shared index-opening helpers out of `cli.py` into `index_folder.py` (§2).
2. ✅ Add `collection_keys` metadata plus `collections.json` in `index`, with the
   backfill done by every run (§4.1, §4.3.2). The Chroma encoding is a native list
   with `$contains` (§4.3.1).
3. ✅ Add scope filtering to retrieval and `answer_question` (§4.3.1).
4. ✅ Add the streaming `answer_question` variant (§6.1).
5. ✅ Build the Textual app: layout, tree, transcript, prompt, history, status bar, keys
   (§3–§8). See §11.1 for how it was built.
6. ✅ Tests (§10), and update PRD.md §6, §7.2 and README.

### 11.1 Step 5 as built

- **Modules** (`src/zotero_rag/tui/`):
  - `main.py`: the `zotero-rag-ui` Typer entry point, `zotero_rag.tui.main:app`. It
    runs the index checks before Textual is imported or the screen is drawn.
  - `session.py`: `QuerySession`, `QuerySettings` and `open_session`.
  - `scopes.py`: snapshot → tree of `ScopeChoice(path, keys)`.
  - `history.py`: `PromptHistory`.
  - `widgets.py` and `app.py`, plus `app.tcss`.
- **Root scope:** the root always searches with no filter (`keys=None`). For a
  whole-library index with several top-level collections, a synthetic `Library` root
  holds them.
- **Notices** from `open_index` (fingerprint adoption, advisory mismatches) appear as
  the first lines of the transcript, because the screen is not up when they arise.
- **Help** is on **F1** everywhere, and on **?** outside the prompt, where `?` has to
  stay typeable. Textual's command palette is disabled.
- **Streaming:** text deltas are queued by the worker thread and drawn by a 0.1 s
  timer on the UI thread, appending to a `Markdown` widget (≤10 redraws/s). Sources
  are drawn after the stream ends.
- **Hybrid mode:** the BM25 index is built on a worker after the screen appears. A
  question sent before it is ready waits for it, and the status bar shows that it is
  waiting. If the build fails, the error is shown on the question.
- **Logging:** with `-v`, logs go to `.zotero-rag-ui/ui.log`. Without it, a
  `NullHandler` keeps Python's last-resort handler from writing to the screen.
- `init`'s `.gitignore` now lists `.zotero-rag-ui/`.

## 12. Acceptance criteria

1. `zotero-rag-ui` run in a folder indexed by this version shows the pinned subtree with
   the root selected and focus on the prompt.
2. A question asked with the root selected gives the same sources as `zotero-rag
   query` with the same flags.
3. With a sub-folder selected, every cited source belongs to an item in that
   folder's subtree.
4. The answer streams into the transcript as Markdown, and the cited Sources list
   follows it.
5. Earlier prompts can be recalled with ↑, edited and re-sent, and they are still
   there after restarting the UI in the same folder.
6. In a folder indexed before this change, the UI starts, answers questions over the
   whole store, and points the user to `zotero-rag index`. After one run, folder
   filtering works with no re-embedding.
7. Run outside an index folder, it exits with the same error message and code as
   `zotero-rag query`.

### 12.1 Verification

All seven criteria are covered by automated tests, run headless with a real Chroma
store and fake embedder and LLM. Criterion 4 (smooth streaming in a real terminal
against OpenRouter) was also checked by hand on 2026-10-28.

| # | test |
|---|---|
| 1 | `test_tui.py::test_launch_selects_the_root_and_focuses_the_prompt` |
| 2 | `test_tui_entry.py::test_the_root_scope_gives_the_same_sources_as_query` (vector, and hybrid with a window) |
| 3 | `test_tui_entry.py::test_a_sub_folder_cites_only_its_own_items`, `test_tui.py::test_a_selected_folder_limits_the_search_to_its_subtree` |
| 4 | `test_tui.py::test_the_answer_is_drawn_while_it_streams_and_completed_at_the_end` |
| 5 | `test_tui.py::test_history_survives_a_restart_and_a_recalled_prompt_can_be_edited` |
| 6 | `test_tui.py::test_without_a_snapshot_the_tree_is_the_root_and_questions_still_work`, `test_cli.py::test_index_backfills_a_store_written_before_membership_existed` |
| 7 | `test_tui_entry.py::test_it_refuses_exactly_as_query_does_and_draws_nothing` (outside a folder, never indexed, wrong embedding model, missing key) |
