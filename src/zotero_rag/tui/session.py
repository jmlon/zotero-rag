"""One UI session's query machinery, built once at launch (PRD-CursesUI §5).

The app talks only to `QuerySession`, so its tests can hand it a session whose
embedder and chat model are local fakes, exactly as `test_querying` does for the
pipeline itself.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from langchain_core.messages import AIMessageChunk, BaseMessage

from zotero_rag.index_folder import Notice, build_keyword_index, open_index, resolve_reranker
from zotero_rag.snapshot import CollectionSnapshot, read_snapshot
from zotero_rag_core.embedding import build_embeddings
from zotero_rag_core.errors import ZoteroRagError
from zotero_rag_core.llm import build_chat
from zotero_rag_core.querying import (
    QueryResult,
    QuestionEmbedder,
    StreamingAnswerer,
    TokenSink,
    stream_answer_question,
)
from zotero_rag_core.reranking import RerankFn
from zotero_rag_core.retrieval import KeywordSearcher, RetrievalMode
from zotero_rag_core.store import ChunkStore, Filters

logger = logging.getLogger(__name__)

Phase = Literal["waiting", "retrieving", "answering"]
PhaseFn = Callable[[Phase], None]


@dataclass(frozen=True)
class QuerySettings:
    """The retrieval settings fixed for the session, from config and launch flags."""

    top_k: int
    window: int
    mode: RetrievalMode
    reranker: str | None
    rerank_fanout: int
    model: str
    #: Most hits per Zotero item (§11.4); 0 is uncapped. Last, with a default, so
    #: settings built before it existed still construct.
    max_per_document: int = 0

    def summary(self) -> str:
        """The status-bar form: `hybrid · top_k 10 · window 1 · rerank cohere · model`."""
        parts = [self.mode, f"top_k {self.top_k}"]
        if self.window:
            parts.append(f"window {self.window}")
        if self.max_per_document:
            parts.append(f"≤{self.max_per_document} per document")
        if self.reranker:
            parts.append(f"rerank {self.reranker}")
        parts.append(self.model)
        return " · ".join(parts)


class QuerySession:
    """Answers questions against one open store with the session's settings.

    The BM25 index for hybrid retrieval is a pass over every chunk, so it is built
    by `build_keyword_index` on a worker after the screen is up (§9). A question
    asked before it is ready waits for it in `ask`, which also runs on a worker.
    """

    def __init__(
        self,
        *,
        store: ChunkStore,
        settings: QuerySettings,
        embed_query: QuestionEmbedder,
        stream: StreamingAnswerer,
        rerank: RerankFn | None = None,
        snapshot: CollectionSnapshot | None = None,
        root_name: str | None = None,
        index_dir: Path | None = None,
        notices: list[Notice] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.snapshot = snapshot
        self.root_name = root_name
        self.index_dir = index_dir
        self.notices = list(notices or [])
        self._embed_query = embed_query
        self._stream = stream
        self._rerank = rerank
        self._keyword_index: KeywordSearcher | None = None
        self._keyword_error: Exception | None = None
        self._keyword_ready = threading.Event()
        if not self.needs_keyword_index:
            self._keyword_ready.set()

    @property
    def needs_keyword_index(self) -> bool:
        return self.settings.mode == "hybrid"

    @property
    def keyword_index_ready(self) -> bool:
        return self._keyword_ready.is_set()

    def build_keyword_index(self) -> None:
        """Build the BM25 index (blocking). Safe to call when it is not needed."""
        if self._keyword_ready.is_set():
            return
        try:
            self._keyword_index = build_keyword_index(self.store, self.settings.mode)
        except Exception as exc:  # noqa: BLE001 - reported on the next question instead
            logger.exception("building the keyword index failed")
            self._keyword_error = exc
        finally:
            self._keyword_ready.set()

    def ask(
        self,
        question: str,
        scope: Collection[str] | None,
        on_token: TokenSink,
        on_phase: PhaseFn,
    ) -> QueryResult:
        """Answer one question (blocking), streaming its text to `on_token`.

        `scope` holds the collection keys of the selected folder and everything
        below it; `None` searches the whole collection.
        """
        if not self._keyword_ready.is_set():
            on_phase("waiting")
            self._keyword_ready.wait()
        if self._keyword_error is not None:
            raise ZoteroRagError(f"The keyword index could not be built: {self._keyword_error}")

        on_phase("retrieving")

        def stream(messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
            # Called once retrieval is done and the prompt is built.
            on_phase("answering")
            return self._stream(messages)

        settings = self.settings
        return stream_answer_question(
            question,
            store=self.store,
            embed_query=self._embed_query,
            stream=stream,
            on_token=on_token,
            top_k=settings.top_k,
            window=settings.window,
            mode=settings.mode,
            keyword_index=self._keyword_index,
            rerank=self._rerank,
            rerank_fanout=settings.rerank_fanout,
            max_per_document=settings.max_per_document,
            filters=Filters.in_collections(scope),
        )


def open_session(
    top_k: int | None, window: int | None, retrieval: RetrievalMode | None, reranker: str | None
) -> QuerySession:
    """Open the current index folder for the UI, applying the launch flags.

    Runs the same checks as `zotero-rag query` and raises the same errors. It makes
    no network call: the clients are built here but first used by a question.
    Notices are kept for the transcript, because the screen is not up yet.
    """
    notices: list[Notice] = []
    config, store = open_index(notices.append)
    rerank, rerank_name = resolve_reranker(config, reranker)
    embeddings = build_embeddings(config)
    chat = build_chat(config)

    def stream(messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
        # A chat model streams AIMessageChunks; `stream` is typed more loosely.
        return cast(Iterator[AIMessageChunk], chat.stream(messages))

    settings = QuerySettings(
        top_k=top_k if top_k is not None else config.index.top_k,
        window=window if window is not None else config.index.retrieval_window,
        mode=retrieval if retrieval is not None else config.index.retrieval,
        reranker=rerank_name,
        rerank_fanout=config.index.reranker_fanout,
        model=config.llm_model,
        max_per_document=config.index.max_per_document,
    )
    index_dir = config.source_dir
    return QuerySession(
        store=store,
        settings=settings,
        embed_query=embeddings.embed_query,
        stream=stream,
        rerank=rerank,
        snapshot=read_snapshot(index_dir) if index_dir is not None else None,
        root_name=config.index.collection_name,
        index_dir=index_dir,
        notices=notices,
    )
