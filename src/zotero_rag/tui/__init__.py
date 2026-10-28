"""`zotero-rag-ui`: a full-screen terminal UI for querying an index folder.

Specified in PRD-CursesUI.md. The modules split along what can be tested without a
terminal: `history` and `scopes` are plain Python, `session` wires the query
pipeline, and only `widgets` and `app` depend on Textual.
"""

#: Per-folder UI state (prompt history, the `-v` log), beside `chroma/` and `cache/`.
UI_DIRNAME = ".zotero-rag-ui"
