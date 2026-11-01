"""Opening an index folder for querying, shared by the CLI and the terminal UI.

Nothing here prints. What the user should be told without failing (a store being
adopted, a setting that changed harmlessly) is handed to an `on_notice` callback,
so the CLI can write it to stderr and the UI can show it in its own window, where a
stray write to the terminal would corrupt the screen.
"""

from __future__ import annotations

import atexit
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from zotero_rag.config import CONFIG_FILENAME, Config, is_reranker_name, load_config
from zotero_rag_core.errors import (
    ConfigError,
    FingerprintMismatchError,
    NotAnIndexFolderError,
    NotIndexedError,
)
from zotero_rag_core.fingerprint import StoreFingerprint, advisory, blocking
from zotero_rag_core.reranking import RerankFn, build_reranker
from zotero_rag_core.retrieval import BM25Searcher, KeywordSearcher, RetrievalMode
from zotero_rag_core.store import ChunkStore, VectorStoreError
from zotero_rag_core.stores.chroma import CHROMA_DIRNAME, ChromaStore

#: Where `--store` may also be given, so a session can be pointed at one database.
STORE_ENV = "ZOTERO_RAG_STORE"
_POSTGRES_SCHEMES = ("postgres://", "postgresql://")

REINDEX_REMEDY = (
    "Run `zotero-rag index --repair` to re-embed the affected chunks with the "
    "configured model (no downloads), or `index --force` to rebuild from scratch. "
    "Reverting the config change also resolves it."
)


@dataclass(frozen=True)
class Notice:
    """Something worth telling the user that does not stop the command."""

    level: Literal["note", "warning"]
    text: str


NoticeFn = Callable[[Notice], None]


def check_fingerprint(store: ChunkStore, config: Config, remedy: str, on_notice: NoticeFn) -> None:
    """Compare what the store was built with against what the config now asks for.

    A store with no fingerprint predates M4: it is *adopted* rather than rejected,
    recording the current config as fact, because re-embedding a perfectly good
    store just to gain a guard would be a poor trade. The notice says plainly what
    that adoption cannot know.
    """
    intended = StoreFingerprint.intended(config, embedding_dim=store.derived_dimension() or 0)
    recorded = store.read_fingerprint()

    if recorded is None:
        store.write_fingerprint(intended)
        if store.count():
            on_notice(
                Notice(
                    "note",
                    f"this store predates fingerprinting; recording "
                    f"{intended.embedding_model} as the model that built it. "
                    "If you changed embedding_model since indexing, run `index --repair`.",
                )
            )
        return

    mismatches = recorded.compare(intended)
    if fatal := blocking(mismatches):
        raise FingerprintMismatchError([str(mismatch) for mismatch in fatal], remedy)
    for mismatch in advisory(mismatches):
        on_notice(Notice("warning", f"{mismatch}."))


def open_store(index_dir: Path, store_url: str | None, *, create: bool) -> ChunkStore:
    """The folder's own Chroma store, or the PostgreSQL store at `store_url`.

    With `create`, a database gets the tables the store needs (an idempotent DDL
    script); only `index` asks for that, so a query never alters a database.
    """
    if not store_url:
        return ChromaStore(index_dir / CHROMA_DIRNAME)
    if not store_url.startswith(_POSTGRES_SCHEMES):
        raise ConfigError(
            "--store takes a PostgreSQL URL (postgresql://user:password@host/database); "
            "leave it out to use this folder's own store."
        )
    # Imported here: psycopg is only loaded by the runs that use a database.
    from zotero_rag_core.stores.pg import PgStore

    store = PgStore.from_url(store_url, create=create)
    # A command ends with the process; an open pool would complain at shutdown.
    atexit.register(store.close)
    return store


def finish_writes(store: ChunkStore) -> None:
    """Bring a database's keyword statistics up to date after an `index` run.

    A search would refresh stale statistics itself, but then the first question
    after indexing would wait for it. A Chroma store keeps nothing of the kind.
    """
    from zotero_rag_core.stores.pg import PgStore

    if isinstance(store, PgStore) and store.keyword_stats_stale():
        store.refresh_keyword_stats()


def open_index(on_notice: NoticeFn, store_url: str | None = None) -> tuple[Config, ChunkStore]:
    """Load the index config and open its store, or explain what is missing.

    The folder is needed either way, for its `config.yaml`; `store_url` only
    changes where the chunks are.
    """
    config = load_config()
    if config.source_dir is None:
        raise NotAnIndexFolderError(
            f"No {CONFIG_FILENAME} in {Path.cwd()}. Run `zotero-rag init` here first, "
            "or change to an index folder."
        )
    if store_url:
        store = open_store(config.source_dir, store_url, create=False)
        try:
            count = store.count()
        except VectorStoreError as exc:
            raise NotIndexedError(
                f"The database has no index yet ({exc}). Run `zotero-rag index --store URL` first."
            ) from exc
        if count == 0:
            raise NotIndexedError(
                "The database holds no chunks. Run `zotero-rag index --store URL` to populate it."
            )
    else:
        chroma_dir = config.source_dir / CHROMA_DIRNAME
        if not ChromaStore.exists(chroma_dir):
            raise NotIndexedError(
                f"This index folder has no vector store yet ({chroma_dir} is empty). "
                'Run `zotero-rag index --collection "NAME"` (or `--collection-key KEY`) first.'
            )
        store = open_store(config.source_dir, None, create=False)
        if store.count() == 0:
            raise NotIndexedError(
                "The vector store holds no chunks. Run `zotero-rag index` to populate it."
            )
    # A question embedded by a different model than the store was built with cannot
    # retrieve meaningfully, so refuse rather than answer from nonsense neighbours.
    check_fingerprint(store, config, REINDEX_REMEDY, on_notice)
    return config, store


def resolve_reranker(config: Config, override: str | None) -> tuple[RerankFn | None, str | None]:
    """The reranker to use and its name, honouring `--reranker none` as "off".

    Returns `(None, None)` when disabled. Building it asserts the credential it
    needs, so a folder that asks for reranking without a key fails immediately.
    """
    name = config.reranker if override is None else (None if override == "none" else override)
    if name is None:
        return None, None
    if not is_reranker_name(name):
        raise ConfigError(
            f"Unknown reranker {name!r}. Use an OpenRouter reranker id such as "
            "`cohere/rerank-4-fast`, `cohere` for Cohere's own API, or `none` to disable it."
        )
    return build_reranker(config, name), name


def build_keyword_index(store: ChunkStore, mode: RetrievalMode) -> KeywordSearcher | None:
    """The keyword half of hybrid retrieval, or `None` for plain vector retrieval.

    A database searches its own full-text index. Otherwise BM25 is built here, once,
    by the caller rather than inside the query, so a run over many questions pays
    for it once; it is a tokenize pass over every chunk in the store.
    """
    if mode != "hybrid":
        return None
    if not isinstance(store, ChromaStore):
        from zotero_rag_core.stores.pg import PgFullTextSearcher, PgStore

        if isinstance(store, PgStore):
            return PgFullTextSearcher(store)
    return BM25Searcher.from_store(store)
