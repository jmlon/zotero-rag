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

from zotero_rag.chunking import chunk_segments
from zotero_rag.embedding import count_tokens
from zotero_rag.errors import ZoteroRagError
from zotero_rag.extraction import extract_note, extract_segments
from zotero_rag.planning import IndexPlan, PlannedAttachment
from zotero_rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

Downloader = Callable[[str], Path]
Embedder = Callable[[list[str]], list[list[float]]]
ProgressHook = Callable[[PlannedAttachment], None]

# Short text used only to learn a model's vector width before repairing.
DIMENSION_PROBE = "dimension probe"


class AttachmentFailure(BaseModel):
    attachment_key: str
    title: str
    error: str


class IndexStats(BaseModel):
    """Outcome of one indexing pass."""

    planned: int = 0
    indexed: int = 0
    reindexed: int = 0
    already_indexed: int = 0
    no_text: int = 0
    failed: int = 0
    pages: int = 0  # segments extracted: pages, sections, or chapters
    chunks: int = 0
    embedding_tokens: int = 0
    elapsed_s: float = 0.0
    failures: list[AttachmentFailure] = Field(default_factory=list)


class RepairStats(BaseModel):
    """Outcome of re-embedding chunks written by a superseded model."""

    stale: int = 0
    repaired: int = 0
    rebuilt: bool = False
    embedding_tokens: int = 0


@contextmanager
def _timed(stats: IndexStats) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        stats.elapsed_s = time.perf_counter() - start


def chunk_id(attachment_key: str, locator: int, chunk_index: int) -> str:
    """Stable id for a chunk, unique within the store.

    The middle component is the locator ordinal — a page for a PDF, a section for
    a snapshot — so ids stay stable and unique across every format.
    """
    return f"{attachment_key}:{locator}:{chunk_index}"


def needs_reindex(store: VectorStore, attachment: PlannedAttachment) -> bool:
    """True when an already-stored attachment has changed in Zotero since indexing.

    Zotero bumps an attachment's `version` when its file or metadata changes, so a
    stored version lower than the current one means the PDF we embedded is stale.
    A stored version of `None` — chunks written before versions were recorded — is
    treated as current, because re-embedding every old store on upgrade would be a
    worse default than leaving it alone (PRD §11.1).
    """
    stored = store.attachment_version(attachment.attachment_key)
    if stored is None or not attachment.version:
        return False
    return attachment.version > stored


def index_plan(
    plan: IndexPlan,
    store: VectorStore,
    download: Downloader,
    embed: Embedder,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str = "",
    on_start: ProgressHook | None = None,
) -> IndexStats:
    """Index every planned attachment, skipping those already stored and unchanged.

    Per PRD §10, one attachment's failure never aborts the run: it is logged as a
    warning, recorded in `stats.failures`, and the pass continues.
    """
    stats = IndexStats(planned=len(plan.attachments))

    with _timed(stats):
        for attachment in plan.attachments:
            if on_start is not None:
                on_start(attachment)

            replacing = False
            if store.has_attachment(attachment.attachment_key):
                if not needs_reindex(store, attachment):
                    logger.info("%s already indexed; skipping", attachment.attachment_key)
                    stats.already_indexed += 1
                    continue
                # Changed in Zotero: drop the old chunks first, so a shorter
                # replacement file cannot leave orphaned tail pages behind.
                removed = store.delete_by_attachment(attachment.attachment_key)
                logger.info(
                    "%s changed in Zotero; re-indexing (dropped %d stale chunk(s))",
                    attachment.attachment_key,
                    removed,
                )
                replacing = True

            try:
                _index_attachment(
                    attachment,
                    store,
                    download,
                    embed,
                    chunk_size,
                    chunk_overlap,
                    embedding_model,
                    stats,
                )
                if replacing:
                    stats.indexed -= 1
                    stats.reindexed += 1
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
    embedding_model: str,
    stats: IndexStats,
) -> None:
    if attachment.needs_download:
        path = download(attachment.attachment_key)
        segments = extract_segments(path, attachment.document_format)
    else:
        # A note's body travels in the item payload, so there is nothing to fetch.
        segments = extract_note(attachment.note_html)

    if not segments:
        # No OCR yet: a scanned PDF yields nothing to embed, and an empty snapshot
        # is indistinguishable from one.
        stats.no_text += 1
        return

    chunks = chunk_segments(segments, chunk_size, chunk_overlap)
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
        ids=[chunk_id(attachment.attachment_key, c.locator, c.index) for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            attachment.chunk_metadata(c.locator, c.index, embedding_model, c.kind, c.label)
            for c in chunks
        ],
    )

    stats.indexed += 1
    stats.pages += len(segments)
    stats.chunks += len(chunks)
    stats.embedding_tokens += count_tokens(texts)


def find_orphans(store: VectorStore, plan: IndexPlan) -> set[str]:
    """Attachment keys in the store that the collection no longer contains.

    One mechanism covers three cases the PRD listed separately: an item deleted in
    Zotero, an item moved out of the pinned collection, and an attachment whose
    content type stopped qualifying. Because a run always fetches the whole
    collection, the set difference is exact and needs no `since` bookkeeping.
    """
    planned = {attachment.attachment_key for attachment in plan.attachments}
    return store.attachment_keys() - planned


def prune_orphans(store: VectorStore, plan: IndexPlan) -> tuple[int, int]:
    """Delete chunks of attachments no longer in the collection.

    Returns (attachments removed, chunks removed).
    """
    orphans = find_orphans(store, plan)
    chunks = sum(store.delete_by_attachment(key) for key in orphans)
    if orphans:
        logger.info("pruned %d attachment(s), %d chunk(s)", len(orphans), chunks)
    return len(orphans), chunks


def repair_embeddings(
    store: VectorStore,
    embed: Embedder,
    embedding_model: str,
) -> RepairStats:
    """Re-embed chunks that a superseded model wrote, using their stored text.

    No download and no PDF parsing: the chunk text is already in the store, so the
    fix costs embedding calls only. Chunk boundaries are preserved exactly, which
    is why this is the remedy for a changed `embedding_model` but *not* for a
    changed `chunk_size` — that needs `--force`.

    A collection's vector width is fixed once written, so a model of a different
    dimension cannot be upserted in place: the whole store is rebuilt from its own
    text instead.
    """
    stale_ids = store.stale_ids(embedding_model)
    stats = RepairStats(stale=len(stale_ids))
    if not stale_ids:
        return stats

    # One cheap call establishes the new model's width before committing to a
    # strategy; embedding a real chunk here would double-count its tokens.
    dimension = store.derived_dimension()
    new_dimension = len(embed([DIMENSION_PROBE])[0])

    if dimension is not None and dimension != new_dimension:
        logger.info(
            "embedding dimension changed (%d -> %d); rebuilding the store from stored text",
            dimension,
            new_dimension,
        )
        chunks = store.all_chunks()
        store.reset()
        stats.rebuilt = True
    else:
        chunks = store.get_by_ids(stale_ids)

    texts = [chunk.text for chunk in chunks]
    embeddings = embed(texts)
    if len(embeddings) != len(chunks):
        raise ZoteroRagError(
            f"Embedding returned {len(embeddings)} vectors for {len(chunks)} chunks"
        )

    metadatas = [{**chunk.metadata, "embedding_model": embedding_model} for chunk in chunks]
    store.upsert(
        ids=[chunk.id for chunk in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    stats.repaired = len(chunks)
    stats.embedding_tokens = count_tokens(texts)
    return stats
