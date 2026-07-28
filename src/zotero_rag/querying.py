"""The query pipeline: embed → retrieve → prompt → answer, with citations.

Like `indexer`, the two network seams (embedding the question, calling the chat
model) are injected as callables, so the retrieval, citation numbering, and
prompt construction are all testable without any API access.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from zotero_rag.documents import LocatorKind
from zotero_rag.embedding import count_tokens
from zotero_rag.llm import LLMError, message_text, usage_from_message
from zotero_rag.vectorstore import StoredChunk, VectorStore

logger = logging.getLogger(__name__)

QuestionEmbedder = Callable[[str], list[float]]
Answerer = Callable[[list[BaseMessage]], AIMessage]

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

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class QueryResult(BaseModel):
    """The answer, the sources it may cite, and what the call cost."""

    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    chunks_retrieved: int = 0


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


def answer_question(
    question: str,
    store: VectorStore,
    embed_query: QuestionEmbedder,
    answer: Answerer,
    top_k: int,
) -> QueryResult:
    """Retrieve `top_k` chunks for the question and have the LLM answer from them.

    With nothing retrieved, no LLM call is made: an empty context can only
    produce "I don't know", and saying so costs nothing.
    """
    start = time.perf_counter()

    embedding = embed_query(question)
    chunks = store.query(embedding, top_k)
    sources, numbers = collect_sources(chunks)
    usage = Usage(embedding_tokens=count_tokens([question]))

    if not chunks:
        logger.info("no chunks retrieved for %r", question)
        usage.elapsed_s = time.perf_counter() - start
        return QueryResult(question=question, answer=NO_CONTEXT_ANSWER, usage=usage)

    context = build_context(chunks, numbers, sources)
    messages = build_messages(question, context)

    try:
        message = answer(messages)
    except Exception as exc:  # noqa: BLE001 - the client raises assorted API errors
        raise LLMError(f"The language model call failed: {exc}") from exc

    usage.prompt_tokens, usage.completion_tokens = usage_from_message(message)
    usage.elapsed_s = time.perf_counter() - start
    text = message_text(message).strip()

    cited = cited_numbers(text)
    for source in sources:
        source.cited = source.number in cited
    logger.info(
        "answered from %d chunk(s) across %d source(s), %d cited",
        len(chunks),
        len(sources),
        len(cited),
    )

    return QueryResult(
        question=question,
        answer=text,
        sources=sources,
        usage=usage,
        chunks_retrieved=len(chunks),
    )
