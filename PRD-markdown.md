# Converting documents to Markdown for easy indexing

LLMs usually perform better when the input is in plan text format.
For this reason we want to create and store markdown (MD) copies of the papers in the Zotero library.

In order to avoid possible information loss, we plan to follow this protocol:
1. The paper entry has an up-to-date MD? If yes, do nothing.
2. The paper entry has a document (PDF, Word, HTML, etc), but not a Markdown: Create the markdown version of the document, upload and attach to the paper in the Zotero library.
3. Optional: Clean big attachments to reduce online storage costs. Add an optional flag `--shrink-storage MAX_FILESIZE`. When the option is present, check if the attached document (excluding MD) is larger than MAX_FILESIZE. If so, check if there is an attached Web Link that resolves to and the link header response returns the correct mime-type. If we have the markdown transcript and the web link to the original file, we can ask confirmation to the user and delete the file attachment. In any other case, preserve the attached file. Produce a report at the end.

For converting source files (PDF, docx, odt, pptx, etc) we can use a document parsing library.
The proposed solution is to use anydoc:
```sh
npx @firecrawl/anydoc report.docx               # Markdown to stdout
npx @firecrawl/anydoc slides.pptx -o slides.md  # or to a file
```
The python version is used as follows (has not been tested)
```python
# pip install firecrawl-anydoc

import anydoc

# 1. Convert a local file directly from its path
markdown_content = anydoc.to_markdown("report.docx")
print(markdown_content)

# 2. Convert from raw bytes (format auto-detected)
with open("presentation.pptx", "rb") as f:
    file_bytes = f.read()

markdown_from_bytes = anydoc.to_markdown_bytes(file_bytes)

# 3. Convert signature-less text formats (like CSV) from bytes
# Note: CSV requires explicitly passing the format extension string
csv_markdown = anydoc.to_markdown_bytes(file_bytes, "csv")
```

We need if it can be invoked from a python application.
Other alternatives (more limited) are MarkItDown.
We want to avoid GPU-based conversion (such as Docling or MinerU) as these have extra hardware requirements.




## Phase 0 findings — anydoc spike (2026-10-29)

Scripts and how to rerun them: `spike/anydoc/` (papers and outputs stay local, git-ignored).
Sample: 29 attachments drawn at random (seed 42) from the library: 23 PDFs, 3 HTML
snapshots, 1 docx, 1 odt, plus 1 more PDF. The library has no EPUB, PPTX or RTF
attachments, so those formats were only checked against the test fixtures.

### anydoc works from Python

`firecrawl-anydoc==0.2.4` installs as one wheel with no dependencies and runs on the project's
Python 3.14. The API is as documented. The exceptions are typed, and
`NeedsOcrError.pages` lists the pages that need OCR.

| | Result |
|---|---|
| Success | 25/29. Every PDF, docx and odt converted except one PDF (`NeedsOcrError`). All 3 HTML snapshots: `UnsupportedError`, as expected. |
| Speed | median **0.08 s** per paper, max 0.27 s; ~190 pages/s. Not a bottleneck. |
| Memory | ~116 MB peak per process (most of it the interpreter). 307 MB for a 37 MB HTML file. |
| ODT / EPUB | Real ODT and the fixture EPUB convert cleanly, including headings and tables. Without its `mimetype` entry, the minimal test-fixture ODT needs `format="odt"`. |

### Quality

The quality metrics proposed for Phase 1b were computed against the pypdf/office
extractors `index` uses today.

- **20 of 23 PDFs are good:** recall 0.86–0.995 (median 0.954), worst page ≥ 0.80. They come
  out with real headings and tables, and the docx and odt score 0.999 and 1.0.
- **3 of 23 PDFs are bad** (13%), and the metrics catch all three:
  - `FL9A5KC7` (1990 LaTeX): every space is lost ("AlgebraicMethodsforInteractive…"). Recall 0.19.
  - `3TZZFT63` (Elsevier, 2018): spaces lost in most lines. Recall 0.84.
  - `6NCSR3YU` (1991 LaTeX): the abstract is misread as a table and words are split. Recall 0.65.
- **New metric: glued words.** The share of "words" with ≥ 20 letters is ≤ 0.005 on every good
  conversion and 0.17–0.19 on the two missing-space failures. It is the clearest single signal.
- **Recall is limited by the baseline.** pypdf is also poor on old LaTeX PDFs ("Metho ds", "/"), so
  recall alone cannot tell which side failed. That is why the gate combines several metrics.

Provisional thresholds, pending the by-eye grades in `spike/anydoc/out/grades.csv`: recall
≥ 0.85, worst page ≥ 0.70, glued words ≤ 0.02, garbage ≤ 0.005, broken spacing ≤ 0.05.

### Page boundaries (D2)

- `anydoc.to_document()` **refuses PDFs** ("PDF converts directly to Markdown"), so anydoc
  exposes no page structure. Whole-document conversion can't produce page markers.
- **Per-page conversion works:** split with pypdf, convert each page, join with
  `<!-- page N -->`. On 8 PDFs it gave the same recall as whole-document conversion, at 2–3× the
  time (median 0.24 s, max 1.2 s for 142 pages).
- Cost: heading detection is weaker on some papers (one paper went from 43 headings to 21), and
  a table that spans pages is split.
- **Per-page also rescues partially scanned PDFs.** `E2Y8PQAZ` (142-page report) fails
  *entirely* as a whole document because 12 pages need OCR. Those pages turn out to be the cover
  and section dividers, with almost no text. Per page, the rest converts with recall 0.98.

### Other findings

- **MarkItDown is not a useful fallback:** recall median 0.79 vs anydoc's 0.95. It got 0.0 on two
  of the papers anydoc handled, it is ~10× slower, and it emits no headings.
- **Bug in the current downloader (affects `index` today):** pyzotero picks a parser from the
  response Content-Type, and `application/vnd.oasis.opendocument.text` is missing from its table.
  An ODT download therefore fails with a UTF-8 `UnicodeDecodeError`, which `download.py` does not
  catch (it only catches `NETWORK_ERRORS`). A plain authenticated `GET /items/{key}/file`
  works. The same applies to any type missing from pyzotero's table (odp, ods, …).
- **Library facts:**
  - 1,043 attachments: 610 stored PDFs and 156 HTML snapshots, plus 1 docx, 1 odt and 10 plain text.
  - **79 stored attachments were never synced**, so conversion can't reach them.
  - **2 Markdown attachments already exist** (user-provided, D3).
  - **76 items have both a stored file and a link attachment.** That is the upper bound for
    `shrink-storage` before the size, quality, annotation and probe checks.

### Grading and final thresholds (D4)

The anydoc outputs were graded by eye in `spike/anydoc/out/grades.csv`: 22 good,
2 bad (`FL9A5KC7`, `3TZZFT63`). `E2Y8PQAZ` was graded ok: its per-page output reads in very
narrow columns.

One grade contradicted the first metric. `6NCSR3YU` reads well, yet its recall was 0.65. The
words it "missed" are pypdf fragments ("alidate", "approac", "ation"): the baseline is broken,
the Markdown isn't. Recall now counts a baseline word as present when it occurs *inside* the
Markdown text. With that change all three metrics agree with every grade:

| | Good (22) | `FL9A5KC7` (bad) | `3TZZFT63` (bad) |
|---|---|---|---|
| recall | ≥ 0.872 | 1.0 | **0.846** |
| worst page | ≥ 0.821 | 1.0 | **0.789** |
| glued words | ≤ 0.005 | **0.168** | **0.191** |

**Thresholds:** recall ≥ 0.85, worst page ≥ 0.80, glued words ≤ 0.02, garbage ≤ 0.005,
broken spacing ≤ 0.05. They separate this sample perfectly, but the sample has only two bad
conversions. The closest good one (`72M7AJY8`, recall 0.872) sits just above the line. A false
FAIL only keeps a file, so the gate errs in the safe direction.

### D2 decided: PDFs are converted page by page

Each page is converted on its own and preceded by `<!-- page N -->`. A page anydoc refuses
(`NeedsOcrError` or `UnsupportedError`, typically covers and dividers) keeps its marker with no
text and is listed in the provenance. With `--ocr hosted`, only those pages need to be sent to
Firecrawl, not the whole paper.

### Zotero write tests (test account, `spike/anydoc/zotero_writes.py`)

All 13 checks passed:

| Check | Result |
|---|---|
| `key_info()` reports write and file access | ✓ `access.user.write`, `.files` |
| Create a child Markdown attachment with `contentType: text/markdown`, a tag and a provenance `note` in one `upload_attachments` call | ✓ all kept; md5 matches |
| Download it with `zot.file` | ✓ returns bytes |
| Replace the file in place (pass `key` + old `md5`) | ✓ same key; new md5; note and tags survive |
| Replace with a stale md5 (someone edited it) | ✓ refused, `PreConditionFailedError` |
| Metadata update with a stale version | ✓ refused, `PreConditionFailedError` (412) |
| Create an annotation on the PDF; find it | ✓ via `children(pdf_key)` and `items(itemType="annotation")`, but **not** among the paper's children |
| Create a `linked_url` attachment | ✓ |
| Move the PDF to the trash (`deleted: 1`) | ✓ |

Findings that shape the implementation:

- **The server deduplicates files.** An upload whose md5 already exists on the server (here the
  same PDF is in the real library) comes back as `unchanged`, not `success`. The attachment
  still gets the file, so both outcomes mean "uploaded".
- **Trashed items vanish from normal listings:** from `/items` and from their parent's children.
  The file stays downloadable from the trash, and the annotations stay attached. So once
  `shrink-storage` has trashed a source, `convert` no longer sees it. The Markdown's provenance
  (`source_removed`, R14) is the only record of it.
- **Finding annotations:** a single `items(itemType="annotation")` listing, grouped by
  `parentItem`, covers the library. It replaces one `children()` call per candidate.
- **Quota:** the Web API doesn't report storage use, so the trash/quota behaviour (R13) can only
  be checked on zotero.org/settings/storage. Not done yet. It doesn't change any decision,
  because the report already tells the user that storage is freed when the trash is emptied.

### Phase 0 result: go

Every exit criterion is met:

- anydoc converts every non-scanned sample. The quality gate matches the human grades.
- D2 is decided.
- The write calls behave as the plan assumed.
