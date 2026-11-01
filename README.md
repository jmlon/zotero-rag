# zotero-rag

A command line tool for asking questions about your own Zotero library.

You pick one Zotero collection, `zotero-rag` reads its attachments — PDFs, saved web
pages, Word and OpenDocument files, EPUBs, plain text, and your own Zotero notes —
and indexes them into a local vector database. Then you ask questions in natural
language. Answers come from an LLM that is only allowed to use the
retrieved passages, and every claim is cited back to the Zotero item and page it
came from — so you can click through and read the source.

Nothing about your library is stored anywhere but your own machine: the PDFs and
the vector store live in a folder you choose. Text is sent to
[OpenRouter](https://openrouter.ai) to be embedded and answered.

```console
$ zotero-rag query --top-k 2 "¿Cuáles son las tareas previas a la escritura de un paper?"

Las tareas previas a la escritura de un paper incluyen realizar un trabajo
preparatorio que generalmente ya debería haberse completado antes de comenzar a
escribir. Esto puede incluir la recopilación de literatura pertinente y la
elaboración de un esquema simple con los puntos relevantes extraídos de esa
literatura [1].

Sources
  [1] Publish like a pro — Powell, Kendall (2010), p. 2
       https://www.zotero.org/jmlondono/collections/TGXVGXGP/items/JU3XRXRB/collection
```

Questions may be asked in any language, whatever language the papers are in.

For a series of questions there is also **`zotero-rag-ui`**, a full-screen terminal
UI. You pick a sub-folder of the collection to search, and the answers stream in as
you read them (see [The terminal UI](#the-terminal-ui)).

There is also a `zotero-rag eval` command that scores the index against a set of
questions, so "does a bigger `top_k` help?" has a number for an answer instead of
an impression — and `zotero-rag generate-questions` will draft that set from the
papers you have already indexed.

## Requirements

- Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/).
- A **Zotero API key** and your numeric library ID, from
  [zotero.org/settings/keys](https://www.zotero.org/settings/keys).
- An **OpenRouter API key**, from [openrouter.ai/keys](https://openrouter.ai/keys).
  Used for embeddings and for the answer; `list-collections` does not need it.

There is no local model: no PyTorch, no HuggingFace, nothing heavyweight to
install. Embeddings and completions are both remote.

## Install

```bash
git clone https://github.com/jmlon/zotero-rag.git
cd zotero-rag
uv sync
```

To get a `zotero-rag` command available everywhere, install it as a tool:

```bash
uv tool install .          # puts `zotero-rag` and `zotero-rag-ui` in ~/.local/bin
```

If the tool was installed before, add `--reinstall` so uv rebuilds it instead of
reusing the cached build.

That copies the current source, so re-run it after pulling or editing the code —
or install with `uv tool install --editable .` to have the command track your
checkout. To go back to running from the source tree, prefix the commands below
with `uv run`.

## Getting started

### 1. Find the collection you want

```bash
zotero-rag list-collections            # add --depth 2 to keep the tree short
```

This prints your collection hierarchy with each collection's key and item count.
Copy the name or the key of the one you want to index.

### 2. Create an index folder

An **index folder** is a directory holding one collection's index. It is marked by
a `config.yaml`, and `index` and `query` must be run from inside one.

```bash
cd ~/ZoteroRAG            # wherever you keep your indexes
zotero-rag init Writing
cd Writing                # index and query must run from inside the folder
```

That writes three files and contacts nothing:

| file | purpose |
|---|---|
| `config.yaml` | every setting, ready to edit; also records which collection this folder serves |
| `.env` | a template for your credentials — fill it in |
| `.gitignore` | keeps `.env`, `chroma/`, and `cache/` out of version control |

Now open `.env` and fill in `ZOTERO_LIBRARY_ID`, `ZOTERO_API_KEY`, and
`OPENROUTER_API_KEY`. Credentials only ever come from the environment or a `.env`
file; `config.yaml` holds the *names* of those variables, never their values, so it
is safe to commit.

`COHERE_API_KEY` is also in the template but can be left blank — it is only needed for
`reranker: cohere`, which calls Cohere directly. Rerankers on OpenRouter use your
OpenRouter key.

**You don't need a copy of your keys in every index folder.** `.env` files are
layered, nearest first — the index folder, then each parent folder, then `~/.env` —
and a blank entry is ignored rather than treated as empty. So one `.env` beside
your index folders, or a single `~/.env`, covers them all:

```
~/ZoteroRAG/
  .env            ← your real keys, once
  Writing/        ← blank template from `init`; inherits from above
  Networking/
```

An exported shell variable beats every file, and among files the nearest filled-in
value wins.

### 3. Index the collection

Name the collection either way — by **name**, or by the **key** that
`list-collections` printed:

```bash
zotero-rag index --collection Networking            # by name
zotero-rag index --collection "Research/Writing"    # by path, for a subcollection
zotero-rag index --collection-key TGXVGXGP          # by key
```

Names are matched case-insensitively and leading or trailing slashes are ignored,
so `datamanagement/data trust` and `/DataManagement/Data Trust` both resolve. If a
name matches more than one collection at the same level, the tool lists the
candidate keys and stops — that is when you use `--collection-key`.

The first run pins that collection into `config.yaml`; later runs need no argument
and reuse it. Subcollections are included recursively.

```bash
zotero-rag index --dry-run     # list what would be indexed, change nothing
zotero-rag index -v            # add timings and token counts
```

Re-running `index` is cheap and safe: attachments already indexed and unchanged are
skipped, and downloaded files stay in `cache/`, so only new work costs anything.
Each run also records which collections every indexed item belongs to. When an item
has moved between subcollections, the run updates that record without re-embedding.

### What gets indexed

| Zotero attachment | Read as | Cited as |
|---|---|---|
| PDF | pages | `p. 4` |
| Saved web page (snapshot) | sections, split at headings | `§ Methods` |
| Word (`.docx`), OpenDocument (`.odt`) | sections, split at headings | `§ Results` |
| EPUB | chapters, in spine order | `§ Chapter Two` |
| Markdown transcript made by `convert` | pages, as marked from the PDF | `p. 4` |
| Other Markdown | sections, split at headings | `§ Title` |
| Plain text, CSV | Markdown headings, else the whole file | `§ Title` |
| Zotero note | the note | `note` |

When an item has **several** readable attachments — very often a published PDF
*and* a saved web page of the same article — only the most preferred one is
indexed, so the same content is not embedded twice and the copies do not compete in
retrieval. `attachment_preference` sets the order; `index_all_attachments: true`
indexes everything. Two attachments of the *same* format (a paper and its
supplement) are both indexed either way. Notes are always extra, never a
replacement.

Anything else — images, video, archives — is skipped, and the run summary names the
content types it skipped so you can see what you are missing:

```
  formats: 13 pdf, 3 html
     7  another attachment of this item is preferred
         7  text/html
     2  no readable attachment
     1  linked URL (no file)
```

Only files Zotero itself holds can be indexed. An attachment whose file never
finished **syncing** to zotero.org, or a **linked file** that lives only on your
computer, is skipped and listed by paper title, with how to fix it. Turn on file
syncing in Zotero's Settings → Sync, or use Tools → Manage Attachments → Convert
Linked Files to Stored Files, then run `index` again.

### Keeping an index in step with Zotero

```bash
zotero-rag index            # picks up items added or replaced in Zotero
zotero-rag index --prune    # also forget items deleted or moved out
```

A plain re-run notices when a PDF was **replaced** in Zotero — the attachment's
version is stored with its chunks — and re-embeds just that attachment, dropping
its old chunks first so nothing stale survives.

Items **deleted** in Zotero, or moved out of this collection, are reported but kept
until you ask: every run tells you how many chunks belong to attachments the
collection no longer has, and `--prune` removes them. Deleting data during an
ordinary index run would be a poor surprise.

`index` works on four documents at a time — downloading, extracting and embedding
them in parallel — since most of the time goes on waiting for Zotero and the
embedding provider. `--jobs N` changes that; both clients back off when rate
limited, but a large `N` on a free tier mostly buys retries.

Every run also records which sub-folders each item is filed in, which is what the
terminal UI's folder filter uses. Moving an item between folders in Zotero is picked
up by the next `index` without re-embedding anything.

### Markdown transcripts for LLMs

```bash
zotero-rag convert --dry-run   # what would be converted, and why
zotero-rag convert             # attach a Markdown transcript to each paper
zotero-rag convert --reconvert
```

`convert` turns the attachment `index` would read — a PDF, Word or OpenDocument
file, an EPUB, a saved web page — into Markdown and **attaches it to the paper in
Zotero**, beside the original. A transcript is useful well beyond this tool: any LLM
reads it more reliably than a PDF. PDFs are converted page by page, so each page is
marked (`<!-- page 4 -->`), and the file opens with front matter naming the paper,
its authors and the attachment it came from. Conversion runs locally (no GPU); the
original is never modified.

It is the first command that **writes to your library**, so it needs an API key
with write access, asks before writing (`--yes` skips the question), and is careful
about what it touches:

- **Re-running is cheap.** A transcript records the checksum of its source; only
  new papers and replaced files are converted again.
- **Your edits are kept.** A transcript edited in Zotero is left alone — even when
  its PDF changes — unless you pass `--force`.
- **Your own Markdown wins.** A paper with a `.md` file you attached counts as
  converted and gets no second transcript.
- **Each transcript is checked** against the text `index` extracts from the same
  file: missing pages, lost spaces between words, or garbled characters are
  reported. A transcript that fails is still attached, since it is still useful,
  but the run lists it.
- **When anydoc refuses a page** that has a text layer (vector figures can
  look like a scan to it), that page's plain text is used instead, and the
  front matter lists it as a `text_layer_pages` entry.
- **After upgrading**, `convert --reconvert` regenerates transcripts made by an
  older converter; a plain run lists how many there are.
- **Scanned pages** have no text to convert. They are reported; `--ocr hosted`
  sends *just those pages* to [Firecrawl Parse](https://firecrawl.dev) for OCR,
  after asking you (set `FIRECRAWL_API_KEY` for higher limits).

**Indexing the transcripts.** A transcript and the file it came from are the
same document, so `index` reads only one of them, and by default the transcript:
in a measured comparison it answered as well as the PDF for about 13% fewer
prompt tokens. Citations still read `p. 4`, since the transcript keeps the PDF's
page numbers. Papers without a transcript are indexed from their files as before.
Markdown you attached yourself counts as a transcript too, and is preferred
over the item's other files.

To index the originals instead, move `markdown` after the formats it transcribes
in the folder's `config.yaml`:

```yaml
index:
  attachment_preference: [pdf, docx, odt, epub, html, markdown, txt]
```

Either way, the next `index` run embeds the copy now preferred and drops the
other copy's chunks, so no passage is retrieved twice. An index folder created
before this change lists its own `attachment_preference` without `markdown`, and
keeps indexing the originals until `markdown` is added to that list.

`--limit N` and `--only-item KEY` try it on a few papers first; `--report run.csv`
(or `.json`) writes a row per paper with its state, outcome and quality verdict.
Transcripts are tagged `zotero-rag:markdown`, so Zotero's tag selector shows them
all.

### Freeing Zotero storage

```bash
zotero-rag shrink-storage --max-size 20MB --dry-run   # what could go, and why the rest stays
zotero-rag shrink-storage --max-size 20MB
```

Once a paper has a good transcript, its large PDF may be worth keeping only as a
link. `shrink-storage` moves a stored file to the **Zotero trash** only when all
of these hold:

- it is larger than `--max-size` (`500KB`, `20MB`, `1.5GB`; 1 KB = 1024 bytes);
- its transcript is up to date and **passed the quality check**. Markdown you
  attached yourself is checked against the file there and then;
- it has **no annotations**. Deleting a PDF deletes its highlights and comments
  in Zotero, which no transcript replaces;
- the same item has a **Link to URI attachment** that serves the very same file:
  it must answer, with the same content type and the right file signature
  (`%PDF`), and **exactly the same size**. The item's own URL and DOI fields are
  not used.

A link that serves a file of the right kind, but of another size (or of no
stated size), is listed under **Check manually** with both sizes. It may be
another version of the paper, so the file is kept and the decision is yours.

The run lists the files it would move with their total size, and asks once.
Then, for each file:

- **A copy is kept** in `shrink-backup/` in the index folder (`--no-backup`
  skips it).
- **The file moves to the Zotero trash.** Nothing is ever deleted outright, so
  a file can be restored from the trash.
- **The transcript's note records** that its source was removed, and where it
  can be downloaded again.

`convert` then reports that transcript as "source removed by shrink-storage" and
never regenerates it.

**Zotero frees the storage only when the trash is emptied.** Emptying it is
left to you, in the Zotero application; by default Zotero also empties it after
30 days.

`--dry-run` checks everything (it downloads files to measure them, and asks
each link) but moves nothing. `--report shrink.csv` (or `.json`) writes a row
per file: moved, check manually, or kept with the reason. Run it with `-v` to
see every kept file and why.

### 4. Ask questions

```bash
zotero-rag query "What makes an abstract effective?"
zotero-rag query "..." --top-k 10        # retrieve more passages
zotero-rag query "..." --window 2        # also send each hit's neighbours
zotero-rag query "..." --retrieval vector # embeddings only, no keyword search
zotero-rag query "..." --reranker cohere/rerank-4-fast  # re-score the passages against the question
zotero-rag query "..." --show-sources    # print the full text of each passage
zotero-rag query "..." --json            # machine-readable answer/sources/usage
zotero-rag query "..." -v                # prompt and completion token counts
```

The source list shows only the sources the answer actually cited. When the
collection has nothing relevant, the tool says it does not know rather than
inventing an answer — that is the point of the citations.

#### The terminal UI

```bash
zotero-rag-ui                     # from inside an index folder
zotero-rag-ui --top-k 5 --reranker cohere/rerank-4-fast   # same flags as `query`
```

The screen has three panes: the collection's folders on the left, the answers on the
right, and a prompt under them.

- **Choose where to search.** Select a folder (Enter, Space or a click) to search it
  and every folder below it. The root, the collection itself, searches everything,
  exactly as `query` does. The status bar shows the current folder, and each answer
  is tagged with the folder it was asked in.
- **Ask.** Type and press **Enter**. **Alt+Enter** (or Shift+Enter, where your
  terminal sends it) starts a new line. The answer streams in as Markdown, followed by
  the sources it cited.
- **Earlier questions.** **↑** and **↓** step through earlier questions, and they are
  kept per index folder across sessions.
- **Keep what you found.**
  - **Ctrl+S** saves the whole session as Markdown. It asks for a file name,
    suggesting `transcripts/zotero-rag-<date>-<time>.md` in the index folder.
  - **Ctrl+L** copies the latest answer, with its sources, to the clipboard. On
    Linux this works most reliably with `xclip`, `xsel` or `wl-clipboard`
    installed. Without one, the copy relies on your terminal supporting it, and some
    (GNOME Terminal, for instance) do not.
- **Moving around.** **Tab** moves between panes. **F1** lists every key. **Ctrl+Q**
  quits.

The folder tree comes from the last `index` run, so the UI makes no Zotero call and
starts instantly. In a folder last indexed by an older version, it searches the
whole collection and asks you to run `zotero-rag index` once to enable folder
filtering. That run updates the stored folder information without re-embedding
anything.

Each question is answered on its own; earlier questions are not sent as context.
`-v` writes diagnostics to `.zotero-rag-ui/ui.log`, because the screen is in use.

### 5. Measure it, before you tune it

Whether a larger `top_k`, a different chunk size, or another embedding model
actually helps is a question with a number for an answer. You need a set of
questions to measure against — and you can have the tool draft one from the papers
it has already indexed:

```bash
zotero-rag generate-questions --dry-run    # what it would read, and the token cost
zotero-rag generate-questions              # writes questions.yaml
zotero-rag generate-questions --foce       # Overwrite questions.yaml
zotero-rag generate-questions --language Spanish --per-document 5
zotero-rag generate-questions --only-item TR6MHCJ2   # try one paper first
```

That writes three questions per indexed document, each with a reference answer,
the points a correct answer should make, and — the useful part — the exact Zotero
item the text came from, so retrieval can be scored against the right paper without
you looking any keys up. It will not overwrite an existing set unless you pass
`--force`.

The questions are meant to make retrieval work for its answer. Each of a
document's questions is a different kind — a specific detail, how a component
works, why a design choice was made, a comparison with a baseline, a limitation,
or when something holds — drawn from passages spread across the whole paper, not
just its abstract. None may name the paper's title, its authors, or the name it
coins for its own method or dataset: those make retrieval a keyword lookup. The
prompt forbids it, and a question that does it anyway is dropped and counted
(`-v` lists them), as is one that says "the study" or "the proposed method"
without saying what it is. A couple of spare questions are asked for per document
to make up for those, but a run can still come back short.

**Read the result and edit it.** Every question is written from one document, so the
set measures whether retrieval can find that document; it won't test pulling an
answer together from four papers, and it can't ask something your library doesn't
answer. It is a good floor, not a substitute for the questions you actually care
about. Adding your own to the same file is the point.

Or write one by hand, beside your `config.yaml`:

```yaml
questions:
    - id: fog-node
      question: What distinguishes a fog node from a cloud node?
      expected_points:
          - it sits between the edge devices and the cloud
          - it provides compute close to where the data is produced
      # Optional: the Zotero item a correct answer should rest on. Titles work too
      # (expected_titles), matched case-insensitively as substrings.
      expected_items: [JSIBCSBW]
      # Optional: the sentence(s) that answer it, copied exactly from the paper.
      # Enables the passage-hit check; generated sets fill it in for you.
      expected_evidence:
          - Fog nodes are deployed between end devices and the cloud data centre.
```

```bash
zotero-rag eval                      # table of results plus one summary line
zotero-rag eval --top-k 10           # same questions, more passages — compare
zotero-rag eval --json > run.json    # the full report, for diffing two runs
zotero-rag eval --only fog-node -v   # one question, with the judge's reasoning
zotero-rag eval --jobs 1             # one question at a time (default: 4 at once)
```

Each run also reports what it cost. The headline is the median **cost per query at
list price**, with its middle half (25th–75th percentile): embedding the question,
reranking (when on) and answering, priced at the answering model's published
OpenRouter rate. Beside it is what was actually **charged**, which is often lower:
providers discount input they have recently seen, and an `eval` repeats the same
prompts minutes apart — two identical runs were charged four times apart. Real
queries are rarely repeats, so list price is the figure to compare configurations
by; the line says what share of the input came from the cache. Grading is
evaluation overhead and is reported apart. `compare-embeddings` shows the median
list price in its `$/query list` column, and every figure goes into the
experiment log.

**The judge sees what the system saw.** It grades against the reference answer
*and* the passages the answer was written from, so a true statement from the
paper that the reference happens to leave out is not marked as fabricated, and a
concise answer that gives exactly what was asked is not marked down for leaving
out background. Runs record this as `judge_protocol: reference+passages`; scores
from runs logged before it (`reference`) are not comparable with later ones.
Showing the passages roughly doubles the cost of grading.

Questions are answered and graded four at a time, since nearly all the time goes on
waiting for the two models; the report keeps the file's order either way. Lower
`--jobs` if the LLM provider returns rate-limit errors.

Each run reports four things per question — the grade, how many expected points
were covered, whether an expected item was **retrieved**, and whether the answer
actually **cited** it — and prints the settings that produced them, so two runs are
comparable. The retrieval and citation columns are mechanical rather than
model-judged, which makes them the ones worth trusting when you compare runs:
they separate "retrieval never found the paper" from "retrieval found it and the
answer ignored it", and those have different fixes.

A question with `expected_evidence` also gets a **passage** column: did the retrieved
chunks contain the sentences that answer it? That is stricter than the hit, which
any chunk of the right paper satisfies — in a small library the right paper is
nearly always found, but the passage that holds the answer often is not, and that
is the failure chunk size and `top_k` actually change. Matching ignores case,
punctuation and line breaks, and a quote split across two retrieved chunks still
counts. It is mechanical too, so it is the number to compare chunkings by.

Two ready-made sets live in `questions/` — a 3-question `Writing` smoke test and a
21-question `Networking` set. A 3-question set turned out too small to resolve any
difference at all, so if you write your own, aim for twenty rather than three.

### Choosing an embedding model and chunking

All three — model, chunk size and overlap — are fixed when a store is built, so comparing them means one store per
combination. `compare-embeddings` builds those stores and scores each one with the
same questions, answering model and judge as `eval`:

```bash
zotero-rag compare-embeddings --dry-run                   # variants, stores, estimated cost
zotero-rag compare-embeddings                             # every model in embedding_models
zotero-rag compare-embeddings --model baai/bge-m3 --chunk-size 500 --chunk-size 2000
zotero-rag compare-embeddings --model baai/bge-m3 --chunk-overlap 0 --chunk-overlap 300
zotero-rag compare-embeddings --no-build                  # re-score the stores already built
zotero-rag compare-embeddings --yes --json > models.json
zotero-rag compare-embeddings --no-build --yes            # No rebuilding indexes, approve execution
```

The configured model at the configured `chunk_size` and `chunk_overlap` is the
**baseline**, read from `chroma/` and never rebuilt. Every other combination goes into
`embeddings/<model>/<chunk_size>-<chunk_overlap>/` (the model id with `/` and `:`
turned into `_`),
indexed from the folder's download cache, so nothing is downloaded twice. The stores
are kept: a later run embeds only what changed in Zotero, which makes adding one
more model to a comparison cheap. Without `--chunk-overlap`, each chunk size keeps
the configured overlap *ratio* (200/1000 → 100 for 500), so the overlap isn't a second
variable. Each `--chunk-overlap` you pass is paired with every chunk size instead,
and a pair whose overlap isn't smaller than its chunk is skipped with a warning.

Stores are built in parallel, one worker per embedding model; a model's own
variants are built one after another, so no provider gets more than one request
at a time from the run. `--workers N` caps how many models build at once (`1` is
fully sequential). Ctrl-C stops the workers after the attachment in hand; what they
wrote is kept, and the next run carries on from there.

Scoring is parallel too: `--eval-workers N` (default 4) variants are answered and
graded at once, each one question at a time. They all share the answering and
grading models, so N is the most requests those models get at once; lower it if
the LLM provider returns rate-limit errors. Chat, grading and embedding requests
retry a rate limit with backoff for about 25 seconds before giving up.

The candidates and their prices come from the `embedding_models` list in
`config.yaml` (the packaged defaults ship one). `--model` also accepts a model
not on the list, which is compared without a price. The table's title names the
`top_k`, window, retrieval mode and reranker every row shares. It shows each
variant's score, retrieval hit, citation and point coverage beside its index
cost, plus the answering tokens per question, because at the same `top_k` larger
chunks make every query more expensive. Tokens are counted with OpenAI's
tokenizer, so costs for other providers are estimates. A model the provider
doesn't serve fails on its row and the rest of the comparison carries on.

Stores of the same chunking built from the same documents hold the same chunks, so
the chunk and token counts are one line under the table. If a store disagrees
(usually a baseline indexed before the collection changed) it is named, because its
score was measured on different documents.

Every run is also saved to `reports/compare-embeddings-<date>-<time>.txt` (the
table as printed) and `.json` (every question's result), so runs can be compared
later.

### The experiment log

Every scored configuration — each `eval` run, and each row of a
`compare-embeddings` run — is also appended to an experiment log for its question
set, `reports/experiments-questions-<version_id>.json`. The log is meant for a
notebook: each entry holds the full settings (models, chunk size and overlap,
`top_k`, window, retrieval mode, reranker, temperatures), the summary numbers and
every question's result, so

```python
import json, pandas as pd
runs = json.load(open("reports/experiments-questions-kg-v1.json"))["runs"]
df = pd.json_normalize(runs)                      # one row per run
per_question = pd.json_normalize(runs, "results", ["run_id", ["settings", "chunk_size"]])
```

Rows of one `compare-embeddings` run share a `batch_id`. Pass `--no-record` for a
quick check you don't want logged.

Results on different question sets are not comparable, so each set has its own
log, named by the `version_id:` at the top of `questions.yaml` (generated sets get
one automatically; a set without one is logged under its content fingerprint).
Each entry also records that fingerprint: **when you edit the questions, change
`version_id`** — if you forget, the next run warns that the set has changed under
the same version.

Grade with a different model from the one answering, so no model marks its own
work. Set `judge_model` once under `global.defaults` for every folder, or under
`eval:` for one folder; left `null`, the answering model grades.

### What the measurements actually said

Everything below came out of `eval` on the 21-question set, and is why the defaults
are what they are:

- **`top_k: 10` is the default because it clearly beat 5**: mean score 2.38 vs 1.81,
  retrieval hit 95% vs 86%, for about 1.9× the tokens per query. Set it back to `5`
  if you would rather have the cheaper query.
- **Past 20 it stops paying.** 10 → 20 gains a little; 20 → 30 gains nothing at all
  while costing 50% more.
- **`retrieval: hybrid` is on by default because it won, for free.** Adding a BM25
  keyword search alongside the vector search and merging the two rankings raised the
  share of answers citing the right paper from 81% to **95%**, and found the right
  paper for 9% more questions on a second question set — while costing the same
  tokens, because it changes *which* passages are sent, not how many. It helps most
  where embeddings are weakest: acronyms, author names, exact phrases. Set
  `retrieval: vector` to turn it off, which also skips building the keyword index
  (about 0.2s per 1,000 chunks at start-up).
- **Hybrid wants `top_k: 10` or more.** At `top_k: 5` roughly half the slots go to
  each retriever, so you keep only two or three vector hits — enough that one test
  question lost the passage it needed and honestly answered "I don't know" instead.
- **`retrieval_window` is off by default because it lost.** Sending each hit's
  neighbours does improve answers at `--window 2` — but spending those same tokens
  on a larger `top_k` improves them more. At every budget tested, a plain run beat
  or matched the windowed one. It is kept because a library with smaller chunks or
  longer documents could come out differently, and one `eval --window 2` will tell
  you. The reason it loses is structural: a neighbour is always from a document you
  already retrieved, so windowing cannot help you find a paper you missed.

- **Reranking is the best thing you can turn on, and it makes queries *cheaper*.**
  With a reranker, the tool retrieves four times as many passages as it needs, has a
  cross-encoder score each one against your actual question, and keeps only the best. Measured: `top_k: 5` with reranking matched `top_k: 10` without it on
  score, beat it on point coverage (73.8% vs 68–72%), raised the share of questions
  that found the right paper from 86% to 95% — and sent **half as much text** to the
  answering model. Those numbers are Cohere's Rerank 4 Fast, measured through
  Cohere's own API; `cohere/rerank-4-fast` is the same model through OpenRouter.

The shipped defaults leave reranking off and use `top_k: 10` with hybrid retrieval.
The better and cheaper setting is `top_k: 5` with a reranker. Any id from the
`rerankers` list in `config.yaml` works, called through OpenRouter on the key you
already have; `reranker: cohere` still calls Cohere's own API with `COHERE_API_KEY`.

```yaml
index:
    top_k: 5
    retrieval: hybrid
    reranker: cohere/rerank-4-fast   # or voyageai/rerank-3, qwen/qwen3-reranker-8b, ...
```

A rate-limited rerank request is retried with backoff; one that still fails falls back
to the retrieved order for that question, and `eval` reports how many questions ran
unreranked. An unknown model or a rejected key stops the run instead.

A note on cost, since "cheaper" is easy to over-read: reranking halves the *text sent
to the LLM*, but adds one rerank request per question, billed separately (Cohere per
search, the others per million tokens; see the `rerankers` list). Whether your
total bill goes down depends on the two providers' prices — `eval` now reports the
cost per query, reranking included, so the second study below could answer it.

**Upgrading an existing index folder?** Your `config.yaml` was written when you ran
`init`, and its values win over the built-in defaults, so an older folder keeps
whatever it says — usually `top_k: 5`. To pick up the new default, edit that one line.
Settings your file does not mention at all, such as `retrieval`, are inherited
automatically.

```bash
zotero-rag eval --top-k 10                 # compare against your current setting
zotero-rag eval --retrieval vector         # what hybrid is buying you
zotero-rag eval --window 2                 # is windowing worth it for your library?
zotero-rag eval --top-k 5 --reranker cohere/rerank-4-fast  # better and cheaper
zotero-rag query "..." --top-k 10
```

Note that grading costs tokens too — about three quarters of a run's bill since
the judge reads the retrieved passages — so `eval` reports it as its own figure
rather than folding it into the cost per query.

### A second study: answering model and reranker

A later round of measurements used a different library — 12 papers on knowledge
graphs and LLMs — and the fairer evaluation this README now describes: 36
generated questions (set `kg-v3`) whose expected points cover only what each
question asks and whose answering passage is checked verbatim, graded by
`google/gemini-3.5-flash-lite` with the retrieved passages in view, and priced at
list price. All rows use 1000/200 chunks, `top_k: 7` and hybrid retrieval:

| Answering model | Reranker | Score | Passage hit | Coverage | Cited | $/query (list) | s/question |
|---|---|---|---|---|---|---|---|
| openai/gpt-4o-mini | — | 3.81 | 27/33 | 72% | 89% | $0.00029 | 3.0 |
| openai/gpt-6-luna | — | 4.00 | 27/33 | 79% | 97% | $0.00021 | 3.5 |
| openai/gpt-4o-mini | voyageai/rerank-3-lite | 4.53 | 33/33 | 91% | 94% | $0.00041 | 3.5 |
| **openai/gpt-6-luna** | **voyageai/rerank-3-lite** | **4.83** | **33/33** | **97%** | **100%** | **$0.00033** | 3.7 |

The best configuration found, at about **$0.33 per thousand queries**:

```yaml
global:
    defaults:
        llm_model: openai/gpt-6-luna
index:
    chunk_size: 1000
    chunk_overlap: 200
    top_k: 7
    retrieval: hybrid
    reranker: voyageai/rerank-3-lite
```

- **Reranking is the largest single gain.** Retrieving 28 candidates and keeping
  the best 7 put the answering passage in front of the model for every question
  that has one, against 27 of 33 without it, and raised the score by 0.7–0.8 for
  both models.
- **The stronger model is better and cheaper.** gpt-6-luna leads on score,
  coverage and citation in both configurations, and costs 20–25% less per query
  at list price; it is about half a second slower per question. Its lead without
  the reranker (+0.19) is within the 0.25 that identical runs were seen to differ
  by; with it (+0.31) it is not.
- **Chunking and `top_k` came first.** On the same library, 1000/200 chunks beat
  500/100 and 1500/300 at every amount of text sent, and without a reranker
  `top_k` 7, 10 and 20 scored the same within noise — so 7, the cheapest.
- **How you grade decides the winner.** Graded against the reference answer
  alone, gpt-4o-mini came out ahead: its longer answers repeated more of the
  reference, while gpt-6-luna's concise ones were marked down for leaving out
  background the question never asked for, and for true statements the reference
  happened to omit. Showing the judge the passages and scoping the expected
  points reversed the ranking.

These are single runs on one library of 12 papers; read them as a starting point
and confirm them on yours with `eval`, which logs every run for comparison.

## Configuring an index

Edit the `config.yaml` in the index folder. Useful settings:

| setting | default | what it does |
|---|---|---|
| `top_k` | `10` | passages retrieved per question; `5` is cheaper, `20` is not better |
| `retrieval_window` | `0` | also send N neighbouring chunks either side of each hit |
| `retrieval` | `hybrid` | `vector` (embeddings only) or `hybrid` (also BM25 keyword search) |
| `reranker` | `null` | re-scores retrieved passages against the question: an OpenRouter reranker id from the `rerankers` list (e.g. `cohere/rerank-4-fast`), or `cohere` for Cohere's own API (needs `COHERE_API_KEY`) |
| `reranker_fanout` | `4` | candidates retrieved per final passage, for the reranker to choose from |
| `attachment_preference` | `[markdown, pdf, docx, odt, epub, html, txt]` | which attachment wins when an item has several; also whether a converted paper is indexed from its source or its transcript |
| `index_all_attachments` | `false` | `true` indexes every readable attachment, not just the best |
| `index_notes` | `true` | index your Zotero notes alongside each item's file |
| `chunk_size` / `chunk_overlap` | `1000` / `200` | characters per chunk, and overlap between them |
| `temperature` | `0.0` | answer randomness; leave at 0 for factual work |
| `llm_model` | `openai/gpt-4o-mini` | any OpenRouter chat model slug |
| `embedding_model` | `openai/text-embedding-3-small` | any OpenRouter embedding slug |
| `eval.questions_file` | `questions.yaml` | golden set used by `eval` |
| `eval.judge_model` | `null` | grading model; `null` inherits `global.defaults.judge_model`, then the answering model |
| `eval.questions_per_document` | `3` | questions `generate-questions` writes per document |
| `eval.document_char_budget` | `30000` | characters of each document shown to the generator |

Changing `chunk_size` or `chunk_overlap` only affects newly indexed attachments —
existing chunks keep the shape they were given. `index --force` rebuilds them all.

**Changing `embedding_model` is safe to attempt:** the store records which model
built it, so the next `index` or `query` stops and tells you, rather than mixing
two incompatible vector spaces and quietly returning nonsense.

```bash
zotero-rag index --repair    # re-embed with the new model, reusing stored text
zotero-rag index --force     # rebuild from scratch (also needed after chunk_size)
```

`--repair` is the cheap path: it needs no downloads and no re-parsing, and it keeps
chunk boundaries exactly as they were. Use `--force` when the boundaries themselves
changed.

Settings resolve as **CLI flag → the folder's `config.yaml` → packaged defaults**.

### Keeping the chunks in PostgreSQL

`index`, `query`, `eval` and `generate-questions` accept `--store` with a
PostgreSQL URL, to keep the chunks in a database with the
[pgvector](https://github.com/pgvector/pgvector) extension instead of the folder's
`chroma/`:

```bash
export ZOTERO_RAG_STORE=postgresql://user:password@host/database   # or pass --store
zotero-rag index        # creates the tables on first use
zotero-rag eval         # grades the database's retrieval with this folder's questions
```

The folder still supplies `config.yaml` and the question sets. This is how a shared
deployment such as `zotero-rag-web` is measured with the same `eval` as an index
folder. Keyword search then runs in the database (full-text search over the same
tokens BM25 uses) instead of being rebuilt in memory at start-up.

## What lives in an index folder

```
<index-folder>/
  config.yaml     settings and the pinned collection
  .env            your credentials (never committed)
  questions.yaml  optional golden set for `eval` — worth committing
  chroma/            the vector store
  embeddings/        stores built by `compare-embeddings`, one per model and chunking
  reports/           a copy of every `compare-embeddings` table, as text and JSON,
                     and one experiment log per question set
  cache/             downloaded attachments and transcripts, kept so re-indexing
                     does not re-download
  collections.json   the indexed collection tree, rewritten by every `index` run
  shrink-backup/     copies of the files `shrink-storage` moved to the Zotero trash
  .zotero-rag-ui/    the terminal UI's prompt history (and its -v log)
```

One folder serves one collection. For a second collection, make a second folder.

## Limitations

This is a proof of concept, and deliberately narrow:

- **Linked URLs are not fetched.** An attachment that is just a link to a web page
  is skipped; only files Zotero actually stores are read.
- **No OCR.** A scanned PDF with no extractable text is skipped with a warning.
- **A saved web page is only as good as what was saved.** Snapshots of paywalled
  articles are usually the abstract page, not the full text.
- **Your own library only** — group libraries are not supported yet.
- Retrieval combines vector similarity with BM25 keywords, and optional reranking.
  The terminal UI can limit a question to a folder, but there is no other metadata
  filtering (by year, author, tag) and no diversity constraint. A single well-matched
  paper can therefore occupy every slot and crowd out the one you wanted.
  A question with no distinctive wording ("what methodology did the authors use?") can
  miss its paper no matter how many passages you retrieve — deeper retrieval, keyword
  search and reranking were all measured against one such question and none of them
  found it.
- Reranking cost is not included in the token counts the tool reports; the
  `rerankers` list in `config.yaml` has each reranker's price.
- The keyword index is rebuilt each time the tool starts, so on a very large store
  `query` pays a second or two before answering. `retrieval: vector` avoids it.
- **Everything is remote per call:** no batching or concurrency yet, so a large
  collection indexes at a steady but unhurried pace.

See `PRD.md` for the full specification, the verified acceptance criteria, and the
list of planned milestones.

## Development

```bash
uv sync --all-groups
uv run pytest -q
uv run ruff check . && uv run mypy
```

The test suite makes no network calls and mocks no Zotero API: the download and
embedding seams are injected, and tests run against a real on-disk Chroma store
with hand-built PDFs.

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
with two packages:

| package | path | what it holds |
|---|---|---|
| `zotero-rag` | `src/zotero_rag/` | the CLI, the terminal UI, index folders, `convert` and `shrink-storage` |
| `zotero-rag-core` | `packages/zotero-rag-core/` | the pipeline: Zotero access, planning, extraction, chunking, embedding, retrieval, reranking, querying, indexing and evaluation |

The core has no command line and no terminal UI, so other applications (such as
the planned `zotero-rag-web`) can depend on it alone. `uv sync` installs both,
and the CLI uses the core from the checkout.

## Reference

- [pyzotero](https://pyzotero.readthedocs.io/en/latest/)
- [Pinecone](https://docs.pinecone.io/reference/sdks/python/overview) (planned
  alternative vector store)

## License

MIT — see `LICENSE`.
