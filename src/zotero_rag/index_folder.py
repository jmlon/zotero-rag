"""Opening an index folder for querying, shared by the CLI and the terminal UI.

Nothing here prints. What the user should be told without failing (a store being
adopted, a setting that changed harmlessly) is handed to an `on_notice` callback,
so the CLI can write it to stderr and the UI can show it in its own window, where a
stray write to the terminal would corrupt the screen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from zotero_rag.config import CONFIG_FILENAME, Config, load_config
from zotero_rag.errors import (
    ConfigError,
    FingerprintMismatchError,
    NotAnIndexFolderError,
    NotIndexedError,
)
from zotero_rag.fingerprint import StoreFingerprint, advisory, blocking
from zotero_rag.reranking import RerankFn, build_reranker
from zotero_rag.retrieval import KeywordIndex, RetrievalMode
from zotero_rag.vectorstore import CHROMA_DIRNAME, VectorStore

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


def check_fingerprint(store: VectorStore, config: Config, remedy: str, on_notice: NoticeFn) -> None:
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


def open_index(on_notice: NoticeFn) -> tuple[Config, VectorStore]:
    """Load the index config and open its store, or explain what is missing."""
    config = load_config()
    if config.source_dir is None:
        raise NotAnIndexFolderError(
            f"No {CONFIG_FILENAME} in {Path.cwd()}. Run `zotero-rag init` here first, "
            "or change to an index folder."
        )
    chroma_dir = config.source_dir / CHROMA_DIRNAME
    if not VectorStore.exists(chroma_dir):
        raise NotIndexedError(
            f"This index folder has no vector store yet ({chroma_dir} is empty). "
            'Run `zotero-rag index --collection "NAME"` (or `--collection-key KEY`) first.'
        )
    store = VectorStore(chroma_dir)
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

    Returns `(None, None)` when disabled. Building it asserts the Cohere credential,
    so a folder that asks for reranking without a key fails immediately.
    """
    name = config.reranker if override is None else (None if override == "none" else override)
    if name is None:
        return None, None
    if name != "cohere":
        raise ConfigError(f"Unknown reranker {name!r}. Use `cohere`, or `none` to disable it.")
    return build_reranker(config), name


def build_keyword_index(store: VectorStore, mode: RetrievalMode) -> KeywordIndex | None:
    """The BM25 index, built once, or `None` for plain vector retrieval.

    Built by the caller rather than inside the query so a run over many questions
    pays for it once; it is a tokenize pass over every chunk in the store.
    """
    if mode != "hybrid":
        return None
    return KeywordIndex.from_store(store)
