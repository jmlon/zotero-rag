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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field

from zotero_rag_core.chunking import Chunk, chunk_segments
from zotero_rag_core.embedding import count_tokens
from zotero_rag_core.errors import ZoteroRagError
from zotero_rag_core.extraction import extract_note, extract_segments
from zotero_rag_core.planning import IndexPlan, PlannedAttachment, SkipReason
from zotero_rag_core.store import ChunkStore

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


def needs_reindex(store: ChunkStore, attachment: PlannedAttachment) -> bool:
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
    store: ChunkStore,
    download: Downloader,
    embed: Embedder,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str = "",
    on_start: ProgressHook | None = None,
    workers: int = 1,
) -> IndexStats:
    """Index every planned attachment, skipping those already stored and unchanged.

    Per PRD §10, one attachment's failure never aborts the run: it is logged as a
    warning, recorded in `stats.failures`, and the pass continues.

    With `workers > 1`, attachments are downloaded, extracted and embedded on that
    many threads, since the time goes on waiting for Zotero and the embedding
    provider. The store is only touched from the calling thread: chromadb is not
    safe to write from several. `on_start` runs on the worker threads, and an
    exception it raises stops the run.
    """
    stats = IndexStats(planned=len(plan.attachments))
    # Named in each warning: `compare-embeddings` builds several models at once,
    # and a rate limit is only actionable if you know whose it was.
    embedding_with = f" with {embedding_model}" if embedding_model else ""

    def prepare(attachment: PlannedAttachment) -> _Prepared:
        if on_start is not None:
            on_start(attachment)
        try:
            return _prepare_attachment(attachment, download, embed, chunk_size, chunk_overlap)
        except ZoteroRagError as exc:
            return _Prepared(error=str(exc))
        except Exception as exc:  # noqa: BLE001 - an unexpected error must not abort the run
            return _Prepared(error=f"unexpected error: {exc}")

    with _timed(stats):
        to_index: list[tuple[PlannedAttachment, bool]] = []
        for attachment in plan.attachments:
            replacing = store.has_attachment(attachment.attachment_key)
            if replacing and not needs_reindex(store, attachment):
                logger.info("%s already indexed; skipping", attachment.attachment_key)
                stats.already_indexed += 1
                continue
            to_index.append((attachment, replacing))

        pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="index")
        try:
            futures = {
                pool.submit(prepare, attachment): (attachment, replacing)
                for attachment, replacing in to_index
            }
            for future in as_completed(futures):
                attachment, replacing = futures[future]
                prepared = future.result()
                if prepared.error is not None:
                    logger.warning(
                        "Skipping %r%s: %s", attachment.title, embedding_with, prepared.error
                    )
                    stats.failed += 1
                    stats.failures.append(
                        AttachmentFailure(
                            attachment_key=attachment.attachment_key,
                            title=attachment.title,
                            error=prepared.error,
                        )
                    )
                    continue
                if replacing:
                    # Changed in Zotero: drop the old chunks first, so a shorter
                    # replacement file cannot leave orphaned tail pages behind.
                    removed = store.delete_by_attachment(attachment.attachment_key)
                    logger.info(
                        "%s changed in Zotero; re-indexing (dropped %d stale chunk(s))",
                        attachment.attachment_key,
                        removed,
                    )
                _store_attachment(attachment, prepared, store, embedding_model, stats)
                if replacing and prepared.chunks:
                    stats.indexed -= 1
                    stats.reindexed += 1
        finally:
            # On Ctrl-C or a failing hook, threads finish the attachment in hand
            # and the rest never start; what was stored stays for the next run.
            pool.shutdown(wait=True, cancel_futures=True)

    return stats


class _Prepared(BaseModel):
    """One attachment, ready to store: its chunks and their vectors, or why not."""

    segments: int = 0
    chunks: list[Chunk] = Field(default_factory=list)
    embeddings: list[list[float]] = Field(default_factory=list)
    tokens: int = 0
    error: str | None = None


def _prepare_attachment(
    attachment: PlannedAttachment,
    download: Downloader,
    embed: Embedder,
    chunk_size: int,
    chunk_overlap: int,
) -> _Prepared:
    if attachment.needs_download:
        path = download(attachment.attachment_key)
        segments = extract_segments(path, attachment.document_format)
    else:
        # A note's body travels in the item payload, so there is nothing to fetch.
        segments = extract_note(attachment.note_html)

    # No OCR yet: a scanned PDF yields nothing to embed, and an empty snapshot
    # is indistinguishable from one.
    chunks = chunk_segments(segments, chunk_size, chunk_overlap) if segments else []
    if not chunks:
        return _Prepared(segments=len(segments))

    texts = [chunk.text for chunk in chunks]
    embeddings = embed(texts)
    if len(embeddings) != len(chunks):
        raise ZoteroRagError(
            f"Embedding returned {len(embeddings)} vectors for {len(chunks)} chunks"
        )
    return _Prepared(
        segments=len(segments),
        chunks=chunks,
        embeddings=embeddings,
        tokens=count_tokens(texts),
    )


def _store_attachment(
    attachment: PlannedAttachment,
    prepared: _Prepared,
    store: ChunkStore,
    embedding_model: str,
    stats: IndexStats,
) -> None:
    chunks = prepared.chunks
    if not chunks:
        stats.no_text += 1
        return

    store.add(
        ids=[chunk_id(attachment.attachment_key, c.locator, c.index) for c in chunks],
        embeddings=prepared.embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            attachment.chunk_metadata(c.locator, c.index, embedding_model, c.kind, c.label)
            for c in chunks
        ],
    )

    stats.indexed += 1
    stats.pages += prepared.segments
    stats.chunks += len(chunks)
    stats.embedding_tokens += prepared.tokens


def find_orphans(store: ChunkStore, plan: IndexPlan) -> set[str]:
    """Attachment keys in the store that the collection no longer contains.

    One mechanism covers three cases the PRD listed separately: an item deleted in
    Zotero, an item moved out of the pinned collection, and an attachment whose
    content type stopped qualifying. Because a run always fetches the whole
    collection, the set difference is exact and needs no `since` bookkeeping.
    """
    planned = {attachment.attachment_key for attachment in plan.attachments}
    return store.attachment_keys() - planned


def prune_orphans(store: ChunkStore, plan: IndexPlan) -> tuple[int, int]:
    """Delete chunks of attachments no longer in the collection.

    Returns (attachments removed, chunks removed).
    """
    orphans = find_orphans(store, plan)
    chunks = sum(store.delete_by_attachment(key) for key in orphans)
    if orphans:
        logger.info("pruned %d attachment(s), %d chunk(s)", len(orphans), chunks)
    return len(orphans), chunks


def drop_replaced_copies(store: ChunkStore, plan: IndexPlan) -> tuple[int, int]:
    """Delete chunks of a document's copy that its other copy now replaces.

    When the preference switches a converted document from its PDF to its
    transcript (or back), the old copy's chunks would sit beside the new ones and
    the same passage would be retrieved twice. Unlike `--prune` this loses nothing:
    a copy is dropped only once the copy replacing it is in the store, so a run in
    which the transcript failed to index keeps the PDF's chunks.

    Returns (attachments removed, chunks removed).
    """
    stored = store.attachment_keys()
    replaced = {
        entry.key
        for entry in plan.skipped_for(SkipReason.SAME_DOCUMENT)
        if entry.key in stored and entry.replaced_by in stored
    }
    chunks = sum(store.delete_by_attachment(key) for key in replaced)
    if replaced:
        logger.info("dropped %d replaced copy(ies), %d chunk(s)", len(replaced), chunks)
    return len(replaced), chunks


def refresh_collection_keys(store: ChunkStore, plan: IndexPlan) -> int:
    """Bring each stored attachment's `collection_keys` in line with the plan.

    Moving an item between folders in Zotero does not bump its attachment's version,
    so the incremental skip would leave stale membership behind. Comparing every
    planned attachment against the store catches that. It also backfills stores
    written before the field existed, at the cost of metadata updates only.
    Returns how many attachments were updated.
    """
    stored = store.collection_keys_by_attachment()
    updated = 0
    for attachment in plan.attachments:
        wanted = sorted(attachment.collection_keys)
        if not wanted or attachment.attachment_key not in stored:
            continue
        if stored[attachment.attachment_key] != wanted:
            store.set_collection_keys(attachment.attachment_key, wanted)
            updated += 1
    if updated:
        logger.info("updated collection membership of %d attachment(s)", updated)
    return updated


def repair_embeddings(
    store: ChunkStore,
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
