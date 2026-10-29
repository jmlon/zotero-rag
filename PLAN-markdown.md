# Markdown transcripts: requirements review and implementation plan

Companion to `PRD-markdown.md`. Part 1 reviews the requirements against the
current code and the Zotero Web API; Part 2 is the implementation plan; Part 3
lists the decisions still open, each with a recommendation.

---

## Part 1 — Requirements review

### 1.1 Can anydoc be called from Python? — Yes, on paper

`firecrawl-anydoc` 0.2.4 is on PyPI (MIT). It is a Rust crate with Python
bindings, shipped as `abi3` wheels for Linux (glibc and musl, x86_64 and
aarch64), macOS (x86_64 and arm64) and Windows x86_64, with **no Python
dependencies** and type stubs included. It imports as `anydoc`; the API in the PRD
is correct, plus:

- `anydoc.to_document(data)`, a document model that also carries embedded assets;
- `anydoc.format_from_bytes / format_from_path / format_from_extension`;
- a typed exception hierarchy under `anydoc.ConvertError`: `UnsupportedError`,
  `NeedsOcrError` (with `.pages`), `MalformedError`, `EncryptedError`,
  `ResourceLimitError`, `MissingPartError`, `HostedError`, plus `OSError`;
- conversion releases the GIL, so a thread pool parallelises it.

**Not verified yet:** installing it in a scratch environment was blocked during
this review, so nothing above has been run. Phase 0 is that check.

Formats it reads: doc/docx/docm, ppt/pptx, xls/xlsx/xlsb, odt/ods/odp, rtf, epub,
csv, pdf. CPU only, so it meets the "no GPU" constraint.

### 1.2 Gaps and risks in the PRD

| # | Issue | Why it matters | Proposed resolution |
|---|---|---|---|
| R1 | **"Up-to-date MD" is undefined.** | Step 1 can't be implemented without it, and a wrong rule either reconverts everything on every run or misses replaced PDFs. | Store provenance on the MD attachment (§2.3). An MD is current when its recorded `source_md5` equals the source attachment's current `md5`. Zotero sends `md5` with every stored attachment, so the check needs no download. |
| R2 | **Which document gets converted** when an item has several (PDF + snapshot is common, ~100 items per PRD §11.2). | Converting both produces two MDs of one paper, the duplication M5 worked to avoid. | Reuse `planning.py`'s candidate collection and `attachment_preference`: one MD per *chosen* source attachment. A paper and its supplement, both PDFs, each get one. |
| R3 | **Markdown has no pages.** Indexing a PDF today yields `p. 4` citations (`LocatorKind.PAGE`). | Indexing the MD instead would silently downgrade citations to `§ Heading`. | Emit page markers (`<!-- page 4 -->`) when converting PDFs, and teach `extract_text` to turn them back into PAGE segments. Phase 0 must find out whether anydoc exposes page boundaries (§2.1). |
| R4 | **anydoc does not read HTML**, and the library has ~205 snapshots. | Step 2 says "PDF, Word, HTML, etc." | Convert HTML with the existing `html_text.parse_sections`, rendering each section as `## title` + body. No new dependency. Use `markdownify` only if tables and lists turn out to matter. |
| R5 | **No local OCR.** Scanned PDFs raise `NeedsOcrError`. anydoc's `ocr="hosted"` **uploads the paper to Firecrawl**. | That is a privacy and licensing decision, not a technical default. | Off by default. `convert --ocr hosted` opts in, and only for documents the local pass reported as `NeedsOcr`, after a confirmation naming how many papers will be sent (D6). |
| R6 | **The tool becomes a writer.** Every command today only reads the library. | A read-only API key will fail halfway through a run, and bugs now cost user data. | Check `key_info()` for write access before planning any writes. Every write path gets `--dry-run`. Writes use `If-Unmodified-Since-Version` (pyzotero supports it), so a concurrent edit in the Zotero client is never overwritten. |
| R7 | **User-edited or user-supplied Markdown.** | Regenerating would overwrite a human's fixes, which is the very "information loss" the PRD wants to avoid. | Managed MDs are recognisable by tag + provenance. If the MD's own `md5` differs from the md5 we uploaded, it was edited: report it and leave it alone unless `--force`. A Markdown attachment the user added counts as "already converted" and is never touched (D3). |
| R8 | **Linked files and unsynced files can't be converted.** | The Web API can't serve them (`planning.SkipReason.LINKED_FILE / NOT_IN_STORAGE`). | Report them with the same wording `index` already uses. |
| R9 | **Shrink: file size is not in Zotero's metadata.** | `--shrink-storage MAX_FILESIZE` needs a size for every attachment. | Use the cached file (conversion downloads it anyway). For attachments not cached, download first. A HEAD request on the file URL is not reliable. |
| R10 | **Shrink: "a Web Link that resolves with the correct mime-type" is weak evidence.** Many publishers reject HEAD (403/405), a DOI resolves to an HTML landing page, paywalls answer 200 with HTML, and links rot. | A loose check deletes files that can't be recovered from the link. | Probe with HEAD, fall back to `GET` with `Range: bytes=0-2047`. Follow redirects. Require status 2xx, a matching `Content-Type`, and for PDFs the `%PDF` magic in the returned bytes. The remote size must also equal the local size (D5). On a mismatch the file is kept and reported for the user to check. |
| R11 | **Shrink: deleting a PDF deletes its annotations.** Zotero highlights and comments are child items (`itemType: annotation`) of the PDF attachment. | Silent loss of the user's reading notes, which no Markdown can replace. | Any source with child annotations is ineligible. Check with `zot.children(key)`. |
| R12 | **Shrink: the Markdown is lossy by construction.** Figures, equations and complex tables don't survive. | It conflicts with the PRD's stated goal of avoiding information loss. | Move to the Zotero **trash** (`deleted: 1`, recoverable) rather than deleting through the API (permanent). Save a local backup copy first by default. Require the MD to pass the quality check (§Phase 1b, D4). |
| R13 | **Trash still counts toward quota.** Storage is reclaimed only when the trash is emptied, either by the client's auto-empty (30 days by default) or manually. | Users would expect immediate savings. The report lists every file moved to the trash and says that emptying the trash in the Zotero application is the user's call (D7). No permanent-delete option. **Verify** the quota behaviour against a test account in Phase 0. |
| R14 | **After shrinking, the MD's source is gone.** | Rule R1 would see "no source" and might treat the MD as orphaned or stale. | On trashing, update the MD's provenance with `source_removed` and `source_url`. The planner treats such an MD as permanently current. |
| R15 | **Confirmation and report formats are unspecified.** | — | One summary table with a single y/N prompt, `--yes` to skip it, and `--report PATH` writing JSON or CSV. Format follows `index`'s summary style (§2.6). |
| R16 | **Scope is unspecified:** whole library or a collection? Must it run inside an index folder? | Every command now runs in an index folder and reuses its target and `cache/`. | Same as `index`: run in an index folder, take `--collection/--collection-key`, default to the pinned target, and write into the same `cache/` so `index` never downloads the MD again. |
| R17 | **No acceptance criteria for conversion quality.** anydoc is v0.2, and academic two-column PDFs are the hard case. | Without a measurement it's guesswork whether indexing MD helps or hurts retrieval, and the project's own practice (PRD §11.3) is to measure first. | Phase 0 produces a sample-quality report. Phase 6 runs `eval` on a PDF-built index vs an MD-built index before changing any default. |
| R18 | **Index interaction.** Once an MD is attached, `index` sees a `text/markdown` attachment. That reads as `DocumentFormat.TEXT`, which ranks last, so the PDF still wins. With `index_all_attachments: true`, both are embedded. | Duplication, and the open `# TO FIX: Include markdown with precedence over txt` in PRD §11.2. | Phase 6 adds a `markdown` format and treats a managed MD and its source as one document for selection. |

Minor: "plan text" → "plain text"; "We need if it can be invoked" → "We need to
know if…". The CLI examples use `npx`; the Python package is the one to use, so
there is no Node.js dependency.

---

## Part 2 — Implementation plan

### Phase 0 — Spike (go/no-go, ~2 days) — ✅ done 2026-10-29: go (findings in `PRD-markdown.md`)

Write throwaway scripts in a scratch folder, not in `src/`.

1. Install `firecrawl-anydoc==0.2.4` and confirm the API in §1.1, the exception
   types, and that `import anydoc` works under Python 3.12+ (the `.venv` runs 3.14).
2. **Sample:** ~30 attachments from the real cache (`<index>/cache/`). Include
   two-column IEEE/ACM PDFs, a long thesis, tables, equations, a scanned PDF, an
   encrypted PDF, docx, odt and epub. For each, record time, peak RSS, the
   exception if any, and output length relative to pypdf's text length.
3. **Page boundaries:** inspect `to_document()` for page information. If there is
   none, measure the fallback: split the PDF with pypdf into one-page PDFs and
   convert each page separately (cost, plus damage to tables that cross pages).
4. Read about 10 outputs by eye for reading order, headers and footers, ligatures,
   references and tables. Compare with MarkItDown on the same 10 to keep the
   fallback option honest. Grade each output good or bad, then compute the
   Phase 1b metrics on all ~30. The good/bad grades are what the quality
   thresholds are calibrated against.
5. **Zotero writes**, against a throwaway account or test library: create a child
   attachment with `attachment_simple`, then set `title`, `contentType`, `tags`
   and `note`; replace an attachment's file in place; move an item to the trash
   with `deleted: 1`; read annotation children; check whether trashed files still
   count toward quota.

**Exit criteria:** conversion succeeds on ≥ 90% of non-scanned samples, the
Zotero write calls behave as assumed, and step 3's measurements are enough to
settle D2. Phase 1's page-marker work and Phase 6's PAGE-segment extraction
depend on that decision. Record the findings in `PRD-markdown.md`.

### Phase 1 — Converter layer

New module `src/zotero_rag/markdown.py`:

```python
class ConversionError(ZoteroRagError): ...
class NeedsOcr(ConversionError): pages: list[int]

class Converted(BaseModel):
    markdown: str
    converter: str           # "anydoc 0.2.4"
    pages: int | None        # when known
    empty_pages: list[int]   # PDF pages anydoc refused (scans, dividers)
    warnings: list[str]

def convert(path: Path, document_format: DocumentFormat) -> Converted
```

- Dispatch: PDF, DOCX, ODT and EPUB go to anydoc. HTML goes to `html_text`
  (R4). TEXT is not converted (it's already text).
- Map anydoc's exceptions onto `ConversionError` subclasses so callers never
  import `anydoc`. Use a lazy import, so other commands keep working if the
  wheel is missing on some platform.
- **PDFs are converted page by page (D2).** Split with pypdf, convert each page
  with `anydoc.to_markdown_bytes(page, "pdf")`, and put `<!-- page N -->` before
  each page's content. Phase 0 showed the same recall as whole-document
  conversion at 2–3× the time, which is still ≈ 1 s for 142 pages. anydoc's
  document model refuses PDFs, so this is the only way to get page boundaries.
  A page anydoc refuses (`NeedsOcrError`, or `UnsupportedError` on near-empty
  pages) keeps its marker with no text and is recorded in `Converted.empty_pages`.
  A partially scanned PDF therefore converts, instead of failing whole as it
  does in one piece. Known costs: heading detection is weaker on some papers,
  and tables that span pages are split.
- Normalise the output: collapse runs of blank lines, strip trailing spaces, end
  with exactly one newline. Same input gives byte-identical output, which keeps
  md5 comparisons meaningful.
- Front matter (YAML) built from `planning.PlannedAttachment` fields: title,
  authors, year, doi, `zotero_item`, `source_attachment`, `source_md5`,
  `converter`, `generated` (ISO date). It is useful to any LLM that reads the file.
- Extend `documents.py` with content types for formats only conversion can read
  (pptx, rtf, doc, xlsx). They map to a new marker that `index` still skips,
  since conversion widens coverage without changing what `index` reads today.
- `pyproject.toml`: `"firecrawl-anydoc>=0.2.4,<0.3"`. Pin below the next minor:
  pre-1.0 APIs move.
- **Hosted OCR (D6):** `convert(..., ocr: Literal["off", "hosted"] = "off")`.
  With `hosted`, a document is first converted locally. Only the pages that
  raised `NeedsOcrError` are sent again with `ocr="hosted"`, as single-page
  PDFs, so documents and pages that don't need OCR never leave the machine. The API key comes from an optional
  `global.firecrawl_api_key_env: FIRECRAWL_API_KEY`, following the existing
  `cohere_api_key_env` pattern. anydoc works without a key, at lower limits.
  `HostedError` becomes a `ConversionError` with the service's message.
  `Converted` gains `ocr: bool`.

### Phase 1b — Quality check (D4)

New module `src/zotero_rag/quality.py`. It answers one question: *does the
Markdown contain the document's text?* The check needs an independent reading of
the same file, and the project already has one: `extraction.extract_segments`,
the pypdf/office/HTML extractors `index` uses today. Every metric compares the MD
against that baseline, one segment (a page, for PDFs) at a time, so a dropped page
can't hide inside a good average.

| Metric | Computation | Catches | Starting threshold |
|---|---|---|---|
| **Word recall** | Normalise both texts (NFKC, casefold, split hyphenated line breaks, strip Markdown syntax). Take distinct words of ≥ 4 letters. Recall = share of the baseline's words found in the MD, either as a word or *inside* the MD text. The second case covers pypdf's broken words ("alidate"), which are not losses. Order-independent, so two-column layouts don't penalise it. | Missing content overall | ≥ 0.85 |
| **Worst-page recall** | The same recall per baseline segment, for segments with ≥ 50 words. Report the minimum. | Dropped pages, truncation at the end, a skipped table-heavy page | ≥ 0.80 |
| **Glued words** (added in Phase 0) | Share of "words" with ≥ 20 letters | Lost spaces ("AlgebraicMethodsforInteractive"), anydoc's most common PDF failure | ≤ 0.02 |
| **Garbage ratio** | U+FFFD, `(cid:NN)`, control and private-use characters, as a share of all characters | Broken font encodings | ≤ 0.5% |
| **Letter ratio** | Letters ÷ non-space characters | Binary or symbol soup | ≥ 0.60 |
| **Broken spacing** | Share of single-letter tokens (other than "a", "I" and digits) | "T h e  r e s u l t s" from bad glyph positioning | ≤ 0.05 |
| **OCR completeness** | No `NeedsOcrError` pages left unconverted | Partially scanned PDFs | none |

```python
class QualityVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"            # the failing metric is named in the report
    UNVERIFIED = "unverified"  # no usable baseline to compare against

class QualityReport(BaseModel):
    verdict: QualityVerdict
    recall: float | None
    worst_segment: tuple[int, float] | None   # (page or section, recall)
    garbage_ratio: float
    letter_ratio: float
    broken_spacing: float
    reasons: list[str]

def assess(markdown: str, baseline: list[Segment]) -> QualityReport
```

- **UNVERIFIED** when the baseline has < 200 words, which always happens for
  scans. An MD produced by hosted OCR therefore can't be verified this way. It
  stays in Zotero but is not eligible for shrink. That is deliberate: no
  independent check exists for it.
- **What it can't measure:** figures, equations and table structure. The report
  shows the PDF's embedded image count as information ("12 images not in the
  transcript"), not as a gate.
- **Thresholds were calibrated in Phase 0** against 24 graded conversions (22
  good, 2 bad). The values above separate them perfectly, though with only two
  bad cases the margin is untested; the closest good one has recall 0.872.
  They live in config (`markdown.quality`), so they can be tightened without a
  release. The Phase 0 prototype is `spike/anydoc/run_spike.py::assess`.
- `convert` runs `assess` right after converting, when it has both texts. It
  stores the verdict in the provenance (`quality=pass recall=0.96`) and lists
  FAIL/UNVERIFIED conversions in its report. A low-quality MD is still uploaded,
  since it's still useful to an LLM, but it is flagged. `shrink-storage` trusts
  a stored PASS and re-runs `assess` when no verdict is stored (user-provided MDs).

### Phase 2 — Zotero write layer

Extend `zotero_api.py`, keeping its pattern of catching `NETWORK_ERRORS` and
raising `ZoteroApiError`:

- `require_write_access(zot)`: from `key_info()`, before any write (R6).
- `create_markdown_attachment(zot, parent_key, path, provenance) -> key`: one
  `upload_attachments` call with an `imported_file` template that already
  carries `title`, `contentType: text/markdown`, tag `zotero-rag:markdown` and
  `note` (verified in Phase 0). Treat an upload result of `unchanged` as success:
  the server deduplicates files by md5.
- `replace_attachment_file(zot, attachment, path)` regenerates in place, so the
  MD's key (and any links to it) stay stable.
- `fetch_annotated_attachments(zot) -> set[str]`: one paginated
  `items(itemType="annotation")` listing, collecting each `parentItem`.
  Annotations are children of the PDF, not of the paper, so the item listings
  `convert` already makes don't include them.
- `trash_item(zot, item)`: set `deleted: 1`. Trashed items drop out of `/items`
  and their parent's children, so after a shrink the source is invisible to the
  planner. The MD's `source_removed` provenance (R14) is the only record of it.
- **Download fix (also a bug in `index` today):** `download.ensure_cached`
  calls `zot.file()`. pyzotero picks a parser from the response content type,
  and `application/vnd.oasis.opendocument.text` (and odp, ods, rtf variants…) is
  missing from its table, so the bytes are parsed as JSON and the download fails
  with `UnicodeDecodeError`. Fetch `GET /items/{key}/file` with httpx instead,
  which is verified to work. It's worth a separate fix and release before this
  feature.
- Every update passes the item's `version` (`If-Unmodified-Since-Version`). A 412
  becomes a "changed in Zotero meanwhile, skipped" outcome, not a crash.

**Provenance format** (in the attachment's `note`, readable without downloading):

```
zotero-rag:markdown v1
source=ABCD1234 source_md5=9e1…  md5=3f2…  converter=anydoc 0.2.4
```

When shrink trashes the source, it adds `source_removed=2026-11-02 source_url=https://…`.
Parse it with a strict regex. A note that doesn't parse means "not ours".

### Phase 3 — Conversion planning (pure, no network)

New module `src/zotero_rag/conversion.py`, mirroring `planning.py`: a pure
function over the raw item payloads, so every rule is unit-testable.

```python
class ConversionState(StrEnum):
    CURRENT       = "up to date"
    MISSING       = "no Markdown yet"           # → create
    STALE         = "source changed"            # → regenerate in place
    OUTDATED      = "older converter"           # → regenerate only with --reconvert
    EDITED        = "Markdown edited in Zotero" # → leave; --force overwrites
    USER_PROVIDED = "Markdown attached by the user"  # → counts as converted (D3)
    SOURCE_GONE   = "source removed by shrink"  # → CURRENT, never touched
    UNCONVERTIBLE = "cannot convert"            # linked, unsynced, unsupported

def build_conversion_plan(collections, preference, force, reconvert) -> ConversionPlan
```

- **User-provided Markdown (D3):** a child attachment counts when it has no
  zotero-rag provenance and either a `text/markdown` / `text/x-markdown` content
  type or a `.md` / `.markdown` filename. Zotero often stores `.md` files as
  `text/plain`, so the filename check matters. One such attachment marks every
  convertible source of that item `USER_PROVIDED`: nothing is generated, and
  `--force` doesn't override it. A user MD that is a linked file still counts as
  converted, but can't be read through the API, so it never makes its item
  eligible for shrink.

- Reuse `planning._collect_candidates` and the preference choice (R2). Refactor
  them into a shared helper rather than copying them. Exclude managed MDs from
  candidates. Then pair each candidate with its managed MD through the
  provenance's `source=` field.
- `ConversionPlan` carries `actions` and `skipped`, with the same `skip_counts()`
  and `content_types_for()` shape as `IndexPlan`, so the CLI reuses the
  summary printer.

### Phase 4 — Execution and the `convert` command

`src/zotero_rag/converter_run.py` (the executor) plus the CLI command in `cli.py`:

```
zotero-rag convert [--collection PATH | --collection-key KEY]
                   [--dry-run] [--yes] [--limit N] [--only-item KEY]...
                   [--reconvert] [--force] [--ocr hosted] [--report PATH]
```

`--ocr hosted` (D6): once the local pass is done, list the documents that need
OCR and print a warning that they will be uploaded to Firecrawl Parse
(firecrawl.dev). Ask one y/N question with the count; `--yes` answers it. Then
convert those. `--dry-run` can't know which documents need OCR without
converting, so it only says that the flag is set.

For each action:

1. `download.ensure_cached`, which reuses the index cache.
2. `markdown.convert`, run on a small thread pool (GIL released), where
   `--jobs` defaults to the CPU count capped at 4. Uploads stay sequential,
   because Zotero rate-limits and pyzotero already honours `Backoff`.
3. Write `cache/<source-key>.md`, then upload: create or replace.
4. **Verify:** re-read the attachment and check that Zotero's `md5` equals the
   local md5. Record a mismatch as a failure. Shrink relies on this check.
5. Rename the cached file to `cache/<md-key>.md` so `index` gets a cache hit.

- One failure never aborts the run. Outcomes are converted, regenerated,
  skipped (reason) and failed (reason).
- `--dry-run` prints the plan and stops: no downloads, no writes.
- A whole-library target asks for confirmation, as `index` does.
- Ctrl-C is safe: each attachment is atomic. The worst case is a converted file
  in the cache that was never uploaded, and the next run picks it up.

### Phase 5 — `shrink-storage`

Implemented as a **separate command** rather than a flag on `convert` (D1), so a
destructive action is never a side effect of a routine one.

```
zotero-rag shrink-storage --max-size 20MB [--collection …] [--dry-run] [--yes]
                          [--no-backup] [--report PATH]
```

Eligibility is a pure function in `src/zotero_rag/shrink.py`. Every condition
must hold, and the first that fails is the reported reason:

1. Stored file (not linked), in a format we convert, larger than `--max-size`
   (parse `500KB`, `20MB` and `1.5GB` as binary units). Size comes from the cache
   (R9).
2. It has a Markdown transcript whose quality verdict is PASS (Phase 1b). That
   is either a managed MD in state `CURRENT` that passed upload verification, or
   a stored user-provided MD (D3), which is downloaded and assessed against this
   source. UNVERIFIED counts as a failure.
3. No child annotations (R11).
4. A sibling `linked_url` attachment (same parent item) whose probe passes
   (R10). Only link attachments count. The parent item's `URL` and `DOI` fields
   are ignored (D5b). An item with several link attachments qualifies if any one
   passes, and the report names the link that passed.
5. **Same size (D5):** the remote size equals the local file's size in bytes.
   It comes from `Content-Length` on HEAD, or the total in `Content-Range` on the
   ranged GET. A mismatch, or a server that reports no size, **keeps the file**.
   The report lists it under *Check manually*, with the URL and both sizes. This
   is the one kept reason that asks the user to act, so it comes first.

Then:

- Print a table (title, file, size, link) with the total bytes reclaimed and a
  **single** y/N prompt. `--yes` skips the prompt; `--dry-run` probes and reports
  but never prompts or writes.
- For each confirmed item: copy the file to `<index>/shrink-backup/<key>.<ext>`
  (on by default, `--no-backup` to skip; D8), move the attachment to the Zotero
  trash (D7), then update the MD's provenance (R14). Never delete permanently.
- **Report**, printed and written with `--report PATH` (JSON or CSV by suffix),
  in three parts:
  1. **Moved to trash:** one row per file, with paper title, filename,
     attachment key, size, the web link it can be recovered from, and the backup
     path. Then the total bytes. A closing note says the storage is freed only
     when the trash is emptied in the Zotero application, and that's left to the
     user.
  2. **Check manually:** a link was found but the size differed or was unknown.
  3. **Kept,** grouped by reason: below size, no transcript, quality FAIL or
     UNVERIFIED (naming the failing metric), annotations, no link, probe failed
     (with status or content type), declined at the prompt, and write errors.

URL probing gets its own `httpx.Client`: 10 s timeout, redirects followed, a
descriptive `User-Agent`, and at most 2 concurrent requests per host.

### Phase 6 — Index integration (separate milestone, gated by eval)

1. `DocumentFormat.MARKDOWN`, separate from `TEXT`. This resolves the PRD §11.2
   TO-FIX. `text/markdown` and `text/x-markdown` map to it.
2. `extract_text` / a new `extract_markdown`: page markers become PAGE segments
   (citations keep `p. 4`), otherwise headings become SECTION segments. Strip the
   front matter before chunking.
3. Selection: a managed MD and its source are **one document**. Even with
   `index_all_attachments: true`, only one of the pair is indexed.
4. Experiment: build two index folders over the `Networking` golden set, one
   with `attachment_preference: [pdf, …]` and one with
   `[markdown, pdf, …]`. Run `eval` on both and adopt the MD-first default
   only if it wins (PRD §11.3 practice).
5. The store fingerprint does not change: indexing a different attachment is
   ordinary re-indexing (version or key change).

### Phase 7 — Tests and docs

- **Unit tests:** provenance parse/format round-trip; `build_conversion_plan`
  over hand-built payloads covering every `ConversionState`; shrink eligibility
  for each reason; size parsing; the URL probe via `httpx.MockTransport` (HEAD
  405 → ranged GET, HTML landing page, redirect chain, wrong magic, size
  mismatch, missing size); markdown normalisation; page-marker extraction;
  user-provided MD detection (`text/plain` + `.md`); `quality.assess` on
  synthetic cases (a dropped page, `(cid:` garbage, spaced-out letters, a baseline
  too short to verify); hosted-OCR retry only after `NeedsOcr` (fake converter,
  no network).
- **Converter tests:** small fixture files (`make_docx_bytes`, `make_odt_bytes`
  and `make_epub_bytes` from `tests/conftest.py`, plus a 2-page PDF). Mark them `skipif` when anydoc is
  not installed.
- **Executor tests:** a fake Zotero client that records writes, asserting that
  `--dry-run` writes nothing, that a 412 is reported and skipped, and that a
  verification mismatch fails the item.
- **Manual acceptance** against a test library, never the real one first.
- **Docs:** README sections for `convert` and `shrink-storage` (with the
  annotation and trash caveats); PRD §3 CLI surface, §6 layout
  (`shrink-backup/`), §9 dependencies; module map §14.3.

### Suggested order and size

| Phase | Deliverable | Size | Depends on |
|---|---|---|---|
| 0 | Spike findings in `PRD-markdown.md` | S | — |
| 1 | `markdown.py` (incl. hosted OCR) + tests | S | 0 |
| 1b | `quality.py` + tests (thresholds from Phase 0) | S | 0 |
| 2 | Write layer + provenance | S | 0 |
| 3 | `conversion.py` (pure) + tests | M | 2 |
| 4 | `convert` command | M | 1, 3 |
| 5 | `shrink-storage` | M | 1b, 4 |
| 6 | Index integration + eval | M | 4 |
| 7 | Docs (continuous) | S | — |

Phases 5 and 6 are independent, so either can ship first. I'd ship 6 first:
it's what makes the Markdown pay off for RAG, while 5 is a storage-cost feature
with real data-loss risk.

---

## Part 3 — Decisions

### Decided (2026-10-29)

- **D1. Shrink is a separate `shrink-storage` command**, not a flag on
  conversion. A destructive operation is never a side effect of a routine one.
- **D2. PDFs are converted page by page** with `<!-- page N -->` markers, so
  indexing the MD keeps `p. 4` citations (decided after Phase 0).
- **D3. Markdown the user attaches counts as "already converted".** It is never
  overwritten or duplicated, even with `--force` (Phase 3). It can make a source
  eligible for shrink if it passes the quality check against that source.
- **D4. Quality is checked with programmable metrics** against the existing
  extractors' text: word recall, worst-page recall, garbage, letter ratio,
  broken spacing, glued words and OCR completeness (Phase 1b). Thresholds were
  calibrated on the Phase 0 grades and are kept in config. Only a PASS makes a file eligible for
  shrink.
- **D5. Link checks require the same content type, magic bytes and an identical
  size.** A size mismatch or an unknown size keeps the file and reports it under
  *Check manually*.
- **D5b. A "web link" is only a `linked_url` attachment** (a Link to URI in the
  Zotero application). The item's own `URL` and `DOI` fields do not count.
- **D6. Hosted OCR is available behind `convert --ocr hosted`**, used only for
  documents the local pass couldn't read, after a confirmation (Phase 1, 4).
- **D7. Shrink moves files to the Zotero trash, never deletes them.** The
  report lists every file moved, and emptying the trash is left to the user in
  the Zotero application.
- **D8. Local backups are on by default** (`--no-backup` to skip).

### Still open

- Optional: check on zotero.org/settings/storage whether trashed files still
  count toward quota (R13). It only affects the report's wording.
