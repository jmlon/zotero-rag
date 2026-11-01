"""The query pipeline: embed → retrieve → prompt → answer, with citations.

Like `indexer`, the two network seams (embedding the question, calling the chat
model) are injected as callables, so the retrieval, citation numbering, and
prompt construction are all testable without any API access.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field

from zotero_rag_core.documents import LocatorKind
from zotero_rag_core.embedding import count_tokens
from zotero_rag_core.llm import (
    LLMError,
    cached_tokens_from_message,
    cost_from_message,
    message_text,
    usage_from_message,
)
from zotero_rag_core.reranking import RerankFn, apply_rerank, candidate_count
from zotero_rag_core.retrieval import KeywordSearcher, RetrievalMode, retrieve
from zotero_rag_core.store import ChunkStore, Filters, StoredChunk

logger = logging.getLogger(__name__)

QuestionEmbedder = Callable[[str], list[float]]
Answerer = Callable[[list[BaseMessage]], AIMessage]
StreamingAnswerer = Callable[[list[BaseMessage]], Iterator[AIMessageChunk]]
TokenSink = Callable[[str], None]

# PRD §8.1. Fixed in the PoC; loading a custom prompt from a file is a future
# option. The language instruction matters because the test corpus is queried in
# Spanish while the passages themselves are in English.
SYSTEM_PROMPT = """You are a research assistant answering questions about a \
collection of academic documents.

Rules:
- Answer using ONLY the numbered context passages provided. Never use outside \
knowledge and never speculate.
- If the passages do not contain the answer, say plainly that you do not know.
- Always cite the passages you used with inline markers such as [1] or [2], \
placed right after the statement they support. The numbers refer to the \
numbered passages.
- Be concise.
- Answer in the same language as the question."""

NO_CONTEXT_ANSWER = (
    "I don't know — no indexed passages matched this question, "
    "so there is nothing in this collection to answer it from."
)


class Source(BaseModel):
    """One cited location: a single page of one attachment."""

    number: int
    item_key: str = ""
    attachment_key: str = ""
    title: str = "(untitled)"
    authors: str = ""
    year: str = ""
    doi: str = ""
    collection_name: str = ""
    zotero_url: str = ""
    page: int = 0
    locator_kind: str = str(LocatorKind.PAGE)
    locator_label: str = ""
    distance: float | None = None
    cited: bool = False
    excerpts: list[str] = Field(default_factory=list)

    @property
    def locator(self) -> str:
        """Where in the document this passage sits, phrased for its format.

        Only a PDF has pages. A snapshot or a Word file has sections, and citing a
        heading (`§ Methods`) is both honest and more useful than inventing a page
        number. Chunks written before M5 carry no kind and are all pages.
        """
        if self.locator_kind == LocatorKind.NOTE:
            return "note"
        if self.locator_kind == LocatorKind.PAGE:
            return f"p. {self.page}"
        if self.locator_kind == LocatorKind.PARAGRAPH:
            return f"¶ {self.page}"
        if self.locator_kind == LocatorKind.SECTION:
            return f"§ {self.locator_label}" if self.locator_label else f"§ {self.page}"
        return f"chunk {self.page}"

    @property
    def label(self) -> str:
        """One-line citation: title, authors, year, locator."""
        head = self.title
        if self.authors:
            head = f"{head} — {self.authors}"
        if self.year:
            head = f"{head} ({self.year})"
        return f"{head}, {self.locator}"


class Usage(BaseModel):
    """Token and timing accounting for one query, reported under `-v`."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    elapsed_s: float = 0.0
    #: USD, as the provider reported it; `None` when it did not. The question's
    #: embedding is priced by the caller, from the `embedding_models` catalogue.
    answer_cost: float | None = None
    rerank_cost: float | None = None
    embedding_cost: float | None = None
    #: Input tokens served from the provider's prompt cache, which it charges less
    #: for; and what the answer would cost at list price, without that discount.
    cached_tokens: int = 0
    answer_list_cost: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class QueryResult(BaseModel):
    """The answer, the sources it may cite, and what the call cost."""

    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    #: Passages similarity search returned — the `top_k` hits.
    chunks_retrieved: int = 0
    #: Passages actually put in front of the model. Larger than `chunks_retrieved`
    #: when windowed retrieval pulled in neighbours (§11.4).
    chunks_in_context: int = 0
    #: Candidates retrieved before reranking cut them down (§11.4.3); equal to
    #: `chunks_retrieved` when no reranker ran.
    candidates_considered: int = 0
    #: Whether reranking actually applied. False after a transient provider failure,
    #: so a silently un-reranked run cannot pass for a reranked one.
    reranked: bool = False


_CITATION_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(answer: str) -> set[int]:
    """The source numbers the answer marked inline, e.g. `[1]` or `[2, 3]`.

    Retrieval always returns passages, but an answer of "I don't know" cites
    none of them — telling the two apart is what keeps the printed source list
    from implying support that the answer never claimed.
    """
    numbers: set[int] = set()
    for match in _CITATION_MARKER.finditer(answer):
        numbers.update(int(part) for part in match.group(1).split(","))
    return numbers


def _text(metadata: dict[str, object], key: str, default: str = "") -> str:
    value = metadata.get(key)
    return default if value is None or value == "" else str(value)


def _page(metadata: dict[str, object]) -> int:
    value = metadata.get("page")
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def collect_sources(chunks: list[StoredChunk]) -> tuple[list[Source], list[int]]:
    """Number the retrieved chunks for citation.

    Chunks from the same page of the same attachment become one source, so the
    answer cites a page once even when two of its chunks were retrieved. Returns
    the sources in citation order and, per chunk, the number it belongs to.
    """
    sources: list[Source] = []
    by_location: dict[tuple[str, int], Source] = {}
    numbers: list[int] = []

    for chunk in chunks:
        metadata = chunk.metadata
        page = _page(metadata)
        location = (_text(metadata, "attachment_key"), page)
        source = by_location.get(location)
        if source is None:
            source = Source(
                number=len(sources) + 1,
                item_key=_text(metadata, "item_key"),
                attachment_key=_text(metadata, "attachment_key"),
                title=_text(metadata, "title", "(untitled)"),
                authors=_text(metadata, "authors"),
                year=_text(metadata, "year"),
                doi=_text(metadata, "doi"),
                collection_name=_text(metadata, "collection_name"),
                zotero_url=_text(metadata, "zotero_url"),
                page=page,
                locator_kind=_text(metadata, "locator_kind", str(LocatorKind.PAGE)),
                locator_label=_text(metadata, "locator_label"),
                distance=chunk.distance,
            )
            sources.append(source)
            by_location[location] = source
        source.excerpts.append(chunk.text)
        numbers.append(source.number)

    return sources, numbers


def expand_window(store: ChunkStore, hits: list[StoredChunk], window: int) -> list[StoredChunk]:
    """Add each hit's ±`window` neighbours in reading order (PRD §11.4).

    A 1,000-character chunk can easily cut a claim from the sentence that gives it
    meaning — "this reduces delay by 13.73%" is far less useful without the chunk
    that says what *this* is. Windowing hands the model the neighbourhood of every
    hit instead of the hit alone.

    Neighbours are taken across the whole **attachment**, not just within the page
    or section the hit sits in. That matters more than it sounds: `chunk_overlap`
    already stitches consecutive chunks *inside* one segment together, but segments
    are chunked independently, so a page or section boundary is the one place in the
    store where no overlap exists at all. Restricting the window to a single segment
    would therefore miss the only join that is genuinely unbridged.

    A chunk is emitted once even when two nearby hits both claim it, and the
    retrieved object is preferred over the refetched one so its similarity distance
    survives. `window <= 0` returns the hits untouched, which is the pre-M7 path.

    Cost is one `get()` per distinct attachment among the hits — no embedding calls,
    no new storage. It fetches whole attachments, which is fine at this scale; a
    store with very long documents would want the ids narrowed first.
    """
    if window <= 0 or not hits:
        return list(hits)

    attachments = {_text(hit.metadata, "attachment_key") for hit in hits}
    ordered: dict[str, list[StoredChunk]] = {
        key: store.get_by_attachment(key) for key in attachments if key
    }
    positions = {
        key: {chunk.id: index for index, chunk in enumerate(sequence)}
        for key, sequence in ordered.items()
    }
    # Hits carry the distance; the refetched copies do not.
    by_id = {hit.id: hit for hit in hits}

    expanded: list[StoredChunk] = []
    emitted: set[str] = set()
    for hit in hits:
        key = _text(hit.metadata, "attachment_key")
        sequence = ordered.get(key, [])
        index = positions.get(key, {}).get(hit.id)
        # An unplaceable hit (no attachment key, or a store written oddly) still
        # reaches the model; it simply gets no neighbours.
        neighbourhood = (
            sequence[max(0, index - window) : index + window + 1] if index is not None else [hit]
        )
        for chunk in neighbourhood:
            if chunk.id in emitted:
                continue
            emitted.add(chunk.id)
            expanded.append(by_id.get(chunk.id, chunk))

    logger.info("windowed retrieval: %d hit(s) expanded to %d chunk(s)", len(hits), len(expanded))
    return expanded


def build_context(chunks: list[StoredChunk], numbers: list[int], sources: list[Source]) -> str:
    """Render the retrieved passages as numbered, attributed blocks.

    The citation line is included in each block so the model can tell the
    passages apart and attribute correctly when two sources look similar.
    """
    by_number = {source.number: source for source in sources}
    blocks = [
        f"[{number}] {by_number[number].label}\n{chunk.text.strip()}"
        for chunk, number in zip(chunks, numbers, strict=True)
    ]
    return "\n\n".join(blocks)


def build_messages(question: str, context: str) -> list[BaseMessage]:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context passages:\n{context}\n\nQuestion: {question}"),
    ]


@dataclass
class _Prepared:
    """Everything decided before the model is called, shared by both variants."""

    question: str
    start: float
    candidates: list[StoredChunk]
    hits: list[StoredChunk]
    reranked: bool
    chunks: list[StoredChunk]
    sources: list[Source]
    usage: Usage
    #: `None` when nothing was retrieved, so there is nothing to ask the model.
    messages: list[BaseMessage] | None


def _prepare(
    question: str,
    store: ChunkStore,
    embed_query: QuestionEmbedder,
    top_k: int,
    window: int,
    mode: RetrievalMode,
    keyword_index: KeywordSearcher | None,
    rerank: RerankFn | None,
    rerank_fanout: int,
    filters: Filters | None,
) -> _Prepared:
    """Retrieve wide → rerank → window → number the sources → build the prompt."""
    start = time.perf_counter()

    embedding = embed_query(question)
    depth = candidate_count(top_k, rerank_fanout) if rerank is not None else top_k
    candidates = retrieve(store, embedding, question, depth, mode, keyword_index, filters)
    hits, reranked, rerank_cost = apply_rerank(candidates, question, top_k, rerank)
    chunks = expand_window(store, hits, window)
    sources, numbers = collect_sources(chunks)
    usage = Usage(embedding_tokens=count_tokens([question]), rerank_cost=rerank_cost)

    messages = None
    if hits:
        messages = build_messages(question, build_context(chunks, numbers, sources))
    else:
        logger.info("no chunks retrieved for %r", question)
    return _Prepared(question, start, candidates, hits, reranked, chunks, sources, usage, messages)


def _no_context(prepared: _Prepared) -> QueryResult:
    prepared.usage.elapsed_s = time.perf_counter() - prepared.start
    prepared.usage.answer_cost = prepared.usage.answer_list_cost = 0.0  # no model call
    return QueryResult(question=prepared.question, answer=NO_CONTEXT_ANSWER, usage=prepared.usage)


def _finish(prepared: _Prepared, message: AIMessage) -> QueryResult:
    """Turn the model's reply into the result: citations, usage, counts."""
    usage = prepared.usage
    usage.prompt_tokens, usage.completion_tokens = usage_from_message(message)
    usage.answer_cost = cost_from_message(message)
    usage.cached_tokens = cached_tokens_from_message(message)
    usage.elapsed_s = time.perf_counter() - prepared.start
    text = message_text(message).strip()

    cited = cited_numbers(text)
    for source in prepared.sources:
        source.cited = source.number in cited
    logger.info(
        "answered from %d chunk(s) across %d source(s), %d cited",
        len(prepared.chunks),
        len(prepared.sources),
        len(cited),
    )

    return QueryResult(
        question=prepared.question,
        answer=text,
        sources=prepared.sources,
        usage=usage,
        chunks_retrieved=len(prepared.hits),
        chunks_in_context=len(prepared.chunks),
        candidates_considered=len(prepared.candidates),
        reranked=prepared.reranked,
    )


def answer_question(
    question: str,
    store: ChunkStore,
    embed_query: QuestionEmbedder,
    answer: Answerer,
    top_k: int,
    window: int = 0,
    mode: RetrievalMode = "vector",
    keyword_index: KeywordSearcher | None = None,
    rerank: RerankFn | None = None,
    rerank_fanout: int = 4,
    filters: Filters | None = None,
) -> QueryResult:
    """Retrieve `top_k` chunks for the question and have the LLM answer from them.

    With `window` above zero, each hit arrives with its neighbours (§11.4). A
    neighbour on another page becomes its own numbered source rather than being
    folded into the hit's: the text really is on that page, and citing where it
    actually sits is more honest than attributing it to the page next door.

    With `mode="hybrid"` and a `keyword_index`, BM25 hits are fused with the vector
    hits before the cut (§11.4.2). Windowing then applies to whatever survived, so
    the two levers compose.

    With a `rerank`, `top_k × rerank_fanout` candidates are retrieved and the
    reranker chooses the `top_k` best (§11.4.3). The order of the three stages is
    fixed and deliberate: **retrieve wide → rerank → window**. Windowing last is what
    keeps a hit's neighbours as context rather than letting them compete as
    candidates, and reranking before the cut is what lets it discard rather than
    merely reorder.

    With `filters`, only chunks that pass them are retrieved (by folder, year or
    tag), so the reranker chooses among them and windowing then stays inside their
    attachments (PRD-CursesUI §4.3.1). `None`, the default, searches the whole store.

    With nothing retrieved, no LLM call is made: an empty context can only
    produce "I don't know", and saying so costs nothing.
    """
    prepared = _prepare(
        question,
        store,
        embed_query,
        top_k=top_k,
        window=window,
        mode=mode,
        keyword_index=keyword_index,
        rerank=rerank,
        rerank_fanout=rerank_fanout,
        filters=filters,
    )
    if prepared.messages is None:
        return _no_context(prepared)

    try:
        message = answer(prepared.messages)
    except Exception as exc:  # noqa: BLE001 - the client raises assorted API errors
        raise LLMError(f"The language model call failed: {exc}") from exc
    return _finish(prepared, message)


def _guarded(stream: StreamingAnswerer, messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
    """The model's chunks, with a provider failure at any point raised as `LLMError`.

    A stream can fail after it has started, mid-answer, not only on the first call.
    The guard wraps the provider alone: an exception from the caller's `on_token`
    is raised in the caller's loop, outside this generator, so it is never
    mislabelled as a model failure.
    """
    try:
        yield from stream(messages)
    except Exception as exc:  # noqa: BLE001 - the client raises assorted API errors
        raise LLMError(f"The language model call failed: {exc}") from exc


def stream_answer_question(
    question: str,
    store: ChunkStore,
    embed_query: QuestionEmbedder,
    stream: StreamingAnswerer,
    on_token: TokenSink,
    top_k: int,
    window: int = 0,
    mode: RetrievalMode = "vector",
    keyword_index: KeywordSearcher | None = None,
    rerank: RerankFn | None = None,
    rerank_fanout: int = 4,
    filters: Filters | None = None,
) -> QueryResult:
    """`answer_question`, with the answer's text sent to `on_token` as it arrives.

    Retrieval, the prompt and the returned `QueryResult` are exactly those of
    `answer_question`, because both run the same `_prepare` and `_finish`; only how
    the reply is obtained differs (PRD-CursesUI §6.1). Citations are parsed only
    once the stream ends, so `cited` is known only from the returned result.

    The chunks are summed into one message, which merges their text and the usage
    the provider sends on the last chunk. That chunk arrives only when the client
    asks for it, which `build_chat` does with `stream_usage`; without it the token
    counts are zero, as with any provider that omits usage.

    `on_token` receives the raw deltas, so their concatenation can differ from
    `result.answer` by surrounding whitespace, which the result strips. When nothing
    is retrieved, `NO_CONTEXT_ANSWER` is sent as one delta, so a caller that renders
    only the deltas still shows it.
    """
    prepared = _prepare(
        question,
        store,
        embed_query,
        top_k=top_k,
        window=window,
        mode=mode,
        keyword_index=keyword_index,
        rerank=rerank,
        rerank_fanout=rerank_fanout,
        filters=filters,
    )
    if prepared.messages is None:
        on_token(NO_CONTEXT_ANSWER)
        return _no_context(prepared)

    merged: AIMessageChunk | None = None
    for chunk in _guarded(stream, prepared.messages):
        merged = chunk if merged is None else merged + chunk
        delta = message_text(chunk)
        if delta:
            on_token(delta)
    return _finish(prepared, merged if merged is not None else AIMessageChunk(content=""))
