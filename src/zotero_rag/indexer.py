"""The indexing pass: download → extract → chunk → embed → store.

Downloading and embedding are injected as callables rather than taken as a
pyzotero client and an embedding model. That keeps the orchestration — including
the incremental skip and the per-attachment failure policy — testable without
any network access.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field

from zotero_rag.chunking import chunk_pages
from zotero_rag.embedding import count_tokens
from zotero_rag.errors import ZoteroRagError
from zotero_rag.extraction import extract_pages
from zotero_rag.planning import IndexPlan, PlannedAttachment
from zotero_rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

Downloader = Callable[[str], Path]
Embedder = Callable[[list[str]], list[list[float]]]
ProgressHook = Callable[[PlannedAttachment], None]


class AttachmentFailure(BaseModel):
    attachment_key: str
    title: str
    error: str


class IndexStats(BaseModel):
    """Outcome of one indexing pass."""

    planned: int = 0
    indexed: int = 0
    already_indexed: int = 0
    no_text: int = 0
    failed: int = 0
    pages: int = 0
    chunks: int = 0
    embedding_tokens: int = 0
    elapsed_s: float = 0.0
    failures: list[AttachmentFailure] = Field(default_factory=list)


@contextmanager
def _timed(stats: IndexStats) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        stats.elapsed_s = time.perf_counter() - start


def chunk_id(attachment_key: str, page: int, chunk_index: int) -> str:
    """Stable id for a chunk, unique within the store."""
    return f"{attachment_key}:{page}:{chunk_index}"


def index_plan(
    plan: IndexPlan,
    store: VectorStore,
    download: Downloader,
    embed: Embedder,
    chunk_size: int,
    chunk_overlap: int,
    on_start: ProgressHook | None = None,
) -> IndexStats:
    """Index every planned attachment, skipping those already stored.

    Per PRD §10, one attachment's failure never aborts the run: it is logged as a
    warning, recorded in `stats.failures`, and the pass continues.
    """
    stats = IndexStats(planned=len(plan.attachments))

    with _timed(stats):
        for attachment in plan.attachments:
            if on_start is not None:
                on_start(attachment)

            if store.has_attachment(attachment.attachment_key):
                logger.info("%s already indexed; skipping", attachment.attachment_key)
                stats.already_indexed += 1
                continue

            try:
                _index_attachment(
                    attachment, store, download, embed, chunk_size, chunk_overlap, stats
                )
            except ZoteroRagError as exc:
                logger.warning("Skipping %r: %s", attachment.title, exc)
                stats.failed += 1
                stats.failures.append(
                    AttachmentFailure(
                        attachment_key=attachment.attachment_key,
                        title=attachment.title,
                        error=str(exc),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - an unexpected error must not abort the run
                logger.warning("Skipping %r: unexpected error: %s", attachment.title, exc)
                stats.failed += 1
                stats.failures.append(
                    AttachmentFailure(
                        attachment_key=attachment.attachment_key,
                        title=attachment.title,
                        error=f"unexpected error: {exc}",
                    )
                )

    return stats


def _index_attachment(
    attachment: PlannedAttachment,
    store: VectorStore,
    download: Downloader,
    embed: Embedder,
    chunk_size: int,
    chunk_overlap: int,
    stats: IndexStats,
) -> None:
    pdf_path = download(attachment.attachment_key)
    pages = extract_pages(pdf_path)
    if not pages:
        # No OCR in the PoC: a scan yields nothing to embed.
        stats.no_text += 1
        return

    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    if not chunks:
        stats.no_text += 1
        return

    texts = [chunk.text for chunk in chunks]
    embeddings = embed(texts)
    if len(embeddings) != len(chunks):
        raise ZoteroRagError(
            f"Embedding returned {len(embeddings)} vectors for {len(chunks)} chunks"
        )

    store.add(
        ids=[chunk_id(attachment.attachment_key, c.page, c.index) for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[attachment.chunk_metadata(c.page, c.index) for c in chunks],
    )

    stats.indexed += 1
    stats.pages += len(pages)
    stats.chunks += len(chunks)
    stats.embedding_tokens += count_tokens(texts)
