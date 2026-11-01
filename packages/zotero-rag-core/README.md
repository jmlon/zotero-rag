# zotero-rag-core

The library behind [`zotero-rag`](../../README.md): reading a Zotero library,
planning and extracting its attachments, chunking, embedding, hybrid retrieval,
reranking, and answering questions with citations.

It has no command line and no terminal UI. Applications build a
`zotero_rag_core.config.Config` however suits them and call the pipeline:
the `zotero-rag` CLI loads YAML index folders, and `zotero-rag-web` reads
environment variables.

Extras:

- `chroma` — the persistent Chroma vector store (`zotero_rag_core.vectorstore`).
