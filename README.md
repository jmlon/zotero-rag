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
uv tool install .          # puts `zotero-rag` in ~/.local/bin
```

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

### What gets indexed

| Zotero attachment | Read as | Cited as |
|---|---|---|
| PDF | pages | `p. 4` |
| Saved web page (snapshot) | sections, split at headings | `§ Methods` |
| Word (`.docx`), OpenDocument (`.odt`) | sections, split at headings | `§ Results` |
| EPUB | chapters, in spine order | `§ Chapter Two` |
| Plain text, Markdown, CSV | Markdown headings, else the whole file | `§ Title` |
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

### 4. Ask questions

```bash
zotero-rag query "What makes an abstract effective?"
zotero-rag query "..." --top-k 10        # retrieve more passages
zotero-rag query "..." --show-sources    # print the full text of each passage
zotero-rag query "..." --json            # machine-readable answer/sources/usage
zotero-rag query "..." -v                # prompt and completion token counts
```

The source list shows only the sources the answer actually cited. When the
collection has nothing relevant, the tool says it does not know rather than
inventing an answer — that is the point of the citations.

## Configuring an index

Edit the `config.yaml` in the index folder. Useful settings:

| setting | default | what it does |
|---|---|---|
| `top_k` | `5` | passages retrieved per question |
| `attachment_preference` | `[pdf, docx, odt, epub, html, txt]` | which attachment wins when an item has several |
| `index_all_attachments` | `false` | `true` indexes every readable attachment, not just the best |
| `index_notes` | `true` | index your Zotero notes alongside each item's file |
| `chunk_size` / `chunk_overlap` | `1000` / `200` | characters per chunk, and overlap between them |
| `temperature` | `0.0` | answer randomness; leave at 0 for factual work |
| `llm_model` | `openai/gpt-4o-mini` | any OpenRouter chat model slug |
| `embedding_model` | `openai/text-embedding-3-small` | any OpenRouter embedding slug |

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

## What lives in an index folder

```
<index-folder>/
  config.yaml     settings and the pinned collection
  .env            your credentials (never committed)
  chroma/         the vector store
  cache/          downloaded PDFs, kept so re-indexing does not re-download
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
- Retrieval is plain vector similarity — no reranking, no hybrid search, no
  metadata filters at query time.
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

## Reference

- [pyzotero](https://pyzotero.readthedocs.io/en/latest/)
- [Pinecone](https://docs.pinecone.io/reference/sdks/python/overview) (planned
  alternative vector store)

## License

MIT — see `LICENSE`.
