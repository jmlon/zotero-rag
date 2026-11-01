"""The query pipeline: retrieval, citation numbering, prompt, and usage.

The store is a real on-disk Chroma instance; only the two network seams
(embedding the question, calling the chat model) are local fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from zotero_rag.config import Config, packaged_defaults
from zotero_rag_core.llm import LLMError, build_chat, message_text, usage_from_message
from zotero_rag_core.querying import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    QueryResult,
    answer_question,
    build_context,
    cited_numbers,
    collect_sources,
    expand_window,
    stream_answer_question,
)
from zotero_rag_core.store import StoredChunk
from zotero_rag_core.stores.chroma import ChromaStore

PdfFactory = Callable[[str, list[str]], Path]


def metadata(
    attachment_key: str = "ATT1",
    page: int = 1,
    title: str = "A Paper",
    chunk_index: int = 0,
) -> dict[str, object]:
    return {
        "item_key": "ITEM1",
        "attachment_key": attachment_key,
        "title": title,
        "authors": "Doe, Jane",
        "year": "2021",
        "doi": "10.1/xyz",
        "collection_name": "Writing",
        "zotero_url": "https://www.zotero.org/u/items/ITEM1",
        "page": page,
        "chunk_index": chunk_index,
    }


def chunk(text: str, **kwargs: object) -> StoredChunk:
    meta = metadata(**kwargs)  # type: ignore[arg-type]
    identifier = f"{meta['attachment_key']}:{meta['page']}:{meta['chunk_index']}"
    return StoredChunk(id=identifier, text=text, metadata=meta, distance=0.25)


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(tmp_path / "chroma")


def populate(store: ChromaStore, texts: list[str]) -> None:
    """Store one chunk per text, each on its own page, with 3-dim embeddings."""
    store.add(
        ids=[f"ATT1:{page}:0" for page in range(1, len(texts) + 1)],
        embeddings=[[float(page), 1.0, 0.5] for page in range(1, len(texts) + 1)],
        documents=texts,
        metadatas=[metadata(page=page) for page in range(1, len(texts) + 1)],
    )


def reply(text: str = "Because of the evidence [1].") -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
    )


# --- citation numbering ---------------------------------------------------


def test_each_page_becomes_one_numbered_source() -> None:
    sources, numbers = collect_sources([chunk("first", page=1), chunk("second", page=4)])
    assert [source.number for source in sources] == [1, 2]
    assert [source.page for source in sources] == [1, 4]
    assert numbers == [1, 2]


def test_two_chunks_of_the_same_page_share_one_citation() -> None:
    """A page cited twice would give the reader two identical source entries."""
    sources, numbers = collect_sources(
        [chunk("start", page=3, chunk_index=0), chunk("rest", page=3, chunk_index=1)]
    )
    assert len(sources) == 1
    assert numbers == [1, 1]
    assert sources[0].excerpts == ["start", "rest"]


def test_same_page_number_in_different_attachments_stays_distinct() -> None:
    sources, _ = collect_sources(
        [chunk("a", attachment_key="ATT1", page=2), chunk("b", attachment_key="ATT2", page=2)]
    )
    assert len(sources) == 2


def test_source_label_carries_title_authors_year_and_page() -> None:
    sources, _ = collect_sources([chunk("text", page=7)])
    assert sources[0].label == "A Paper — Doe, Jane (2021), p. 7"


def test_missing_metadata_degrades_instead_of_failing() -> None:
    """Chunks written by an older run may lack fields; citation must still render."""
    bare = StoredChunk(id="X:0:0", text="text", metadata={"attachment_key": "X"})
    sources, numbers = collect_sources([bare])
    assert numbers == [1]
    assert sources[0].title == "(untitled)"
    assert sources[0].page == 0
    assert sources[0].year == ""


def test_context_blocks_are_numbered_and_attributed() -> None:
    chunks = [chunk("First passage", page=1), chunk("Second passage", page=2)]
    sources, numbers = collect_sources(chunks)
    context = build_context(chunks, numbers, sources)
    assert "[1] A Paper — Doe, Jane (2021), p. 1" in context
    assert "First passage" in context
    assert "[2] A Paper — Doe, Jane (2021), p. 2" in context


# --- the pipeline --------------------------------------------------------


def test_answers_from_the_retrieved_chunks(store: ChromaStore) -> None:
    populate(store, ["Writers should write every day.", "Keep previous versions of files."])
    seen: list[list[BaseMessage]] = []

    def ask(messages: list[BaseMessage]) -> AIMessage:
        seen.append(messages)
        return reply()

    result = answer_question(
        "What should writers do?",
        store=store,
        embed_query=lambda question: [1.0, 1.0, 0.5],
        answer=ask,
        top_k=2,
    )

    assert result.answer == "Because of the evidence [1]."
    assert result.chunks_retrieved == 2
    assert len(result.sources) == 2
    # The prompt carries the rules and the retrieved text, and nothing else.
    system, human = seen[0]
    assert str(system.content) == SYSTEM_PROMPT
    assert "What should writers do?" in str(human.content)
    assert "write every day" in str(human.content)


def test_top_k_bounds_the_retrieval(store: ChromaStore) -> None:
    populate(store, [f"passage {number}" for number in range(1, 6)])
    result = answer_question(
        "anything",
        store=store,
        embed_query=lambda question: [1.0, 1.0, 0.5],
        answer=lambda messages: reply(),
        top_k=2,
    )
    assert result.chunks_retrieved == 2


def test_usage_is_reported_for_the_verbose_summary(store: ChromaStore) -> None:
    populate(store, ["some text"])
    result = answer_question(
        "why?",
        store=store,
        embed_query=lambda question: [1.0, 1.0, 0.5],
        answer=lambda messages: reply(),
        top_k=1,
    )
    assert result.usage.prompt_tokens == 120
    assert result.usage.completion_tokens == 30
    assert result.usage.total_tokens == 150
    assert result.usage.embedding_tokens > 0
    assert result.usage.elapsed_s >= 0.0


def test_an_empty_store_says_it_does_not_know_without_calling_the_llm(
    store: ChromaStore,
) -> None:
    """Acceptance criterion 8's floor: no context can only mean "I don't know"."""
    calls: list[int] = []

    result = answer_question(
        "unsupported question",
        store=store,
        embed_query=lambda question: [1.0, 1.0, 0.5],
        answer=lambda messages: calls.append(1) or reply(),  # type: ignore[func-returns-value]
        top_k=5,
    )

    assert calls == []
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []


def test_a_failing_llm_call_becomes_a_user_facing_error(store: ChromaStore) -> None:
    populate(store, ["some text"])

    def ask(messages: list[BaseMessage]) -> AIMessage:
        raise RuntimeError("402 insufficient credits")

    with pytest.raises(LLMError, match="402 insufficient credits"):
        answer_question(
            "why?",
            store=store,
            embed_query=lambda question: [1.0, 1.0, 0.5],
            answer=ask,
            top_k=1,
        )


# --- which sources the answer used ---------------------------------------


def test_inline_markers_are_read_back_out_of_the_answer() -> None:
    assert cited_numbers("Alpha [1] and beta [2, 3]. Also [2].") == {1, 2, 3}


def test_an_answer_with_no_markers_cites_nothing() -> None:
    assert cited_numbers("No sé.") == set()
    assert cited_numbers("See section [A] of the text.") == set()


def test_sources_record_whether_the_answer_cited_them(store: ChromaStore) -> None:
    """Acceptance criterion 8: an "I don't know" must not look supported."""
    populate(store, ["passage one", "passage two"])
    result = answer_question(
        "unrelated question",
        store=store,
        embed_query=lambda question: [1.0, 1.0, 0.5],
        answer=lambda messages: reply("No sé."),
        top_k=2,
    )
    assert result.answer == "No sé."
    assert [source.cited for source in result.sources] == [False, False]


def test_only_the_cited_sources_are_marked(store: ChromaStore) -> None:
    populate(store, ["passage one", "passage two"])
    result = answer_question(
        "why?",
        store=store,
        embed_query=lambda question: [2.0, 1.0, 0.5],
        answer=lambda messages: reply("Only the second matters [2]."),
        top_k=2,
    )
    cited = {source.number for source in result.sources if source.cited}
    assert cited == {2}


# --- llm helpers ---------------------------------------------------------


def test_missing_usage_metadata_reports_zeros() -> None:
    assert usage_from_message(AIMessage(content="hi")) == (0, 0)


def test_block_style_content_is_flattened_into_text() -> None:
    """Some providers return content blocks; the answer must not be a repr."""
    message = AIMessage(
        content=[{"type": "text", "text": "First. "}, {"type": "text", "text": "Second."}]
    )
    assert message_text(message) == "First. Second."


# --- windowed retrieval (M7, §11.4) --------------------------------------


def populate_document(store: ChromaStore, pages: dict[int, list[str]]) -> None:
    """Store a multi-page attachment: {page: [chunk text, ...]}.

    The embedding encodes the page, so a query vector can select a known page.
    """
    ids, embeddings, documents, metadatas = [], [], [], []
    for page, texts in pages.items():
        for index, text in enumerate(texts):
            ids.append(f"ATT1:{page}:{index}")
            embeddings.append([float(page), float(index), 0.5])
            documents.append(text)
            metadatas.append(metadata(page=page, chunk_index=index))
    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def test_a_window_of_zero_is_the_pre_m7_path(store: ChromaStore) -> None:
    populate_document(store, {1: ["a", "b", "c"]})
    hits = [chunk("b", page=1, chunk_index=1)]
    assert expand_window(store, hits, window=0) == hits


def test_a_window_adds_the_neighbours_either_side(store: ChromaStore) -> None:
    populate_document(store, {1: ["first", "second", "third", "fourth"]})
    hits = store.query([1.0, 1.0, 0.5], top_k=1)
    assert [hit.text for hit in hits] == ["second"]

    expanded = expand_window(store, hits, window=1)

    # Reading order, not hit-first order: the model reads it as prose.
    assert [c.text for c in expanded] == ["first", "second", "third"]


def test_a_window_crosses_a_page_boundary(store: ChromaStore) -> None:
    """The one join with no `chunk_overlap` to stitch it: segments are chunked apart."""
    populate_document(store, {1: ["p1c0", "p1c1"], 2: ["p2c0", "p2c1"]})
    hits = [chunk("p2c0", page=2, chunk_index=0)]

    expanded = expand_window(store, hits, window=1)

    assert [c.text for c in expanded] == ["p1c1", "p2c0", "p2c1"]


def test_a_window_stops_at_the_edges_of_the_document(store: ChromaStore) -> None:
    populate_document(store, {1: ["only", "second"]})
    hits = [chunk("only", page=1, chunk_index=0)]
    assert [c.text for c in expand_window(store, hits, window=5)] == ["only", "second"]


def test_overlapping_windows_emit_each_chunk_once(store: ChromaStore) -> None:
    """Two adjacent hits would otherwise repeat the same text in the prompt."""
    populate_document(store, {1: ["a", "b", "c", "d", "e"]})
    hits = [chunk("b", page=1, chunk_index=1), chunk("c", page=1, chunk_index=2)]

    expanded = expand_window(store, hits, window=1)

    assert [c.text for c in expanded] == ["a", "b", "c", "d"]
    assert len({c.id for c in expanded}) == len(expanded)


def test_a_window_does_not_reach_into_another_attachment(store: ChromaStore) -> None:
    """Neighbours are only neighbours within one document."""
    store.add(
        ids=["ATT1:1:0", "ATT2:1:0"],
        embeddings=[[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]],
        documents=["from one", "from two"],
        metadatas=[metadata(attachment_key="ATT1"), metadata(attachment_key="ATT2")],
    )
    hits = [chunk("from one", attachment_key="ATT1", page=1)]
    assert [c.text for c in expand_window(store, hits, window=3)] == ["from one"]


def test_the_hit_keeps_its_similarity_distance(store: ChromaStore) -> None:
    """Refetched neighbours have no distance; the hit's must survive the expansion."""
    populate_document(store, {1: ["a", "b", "c"]})
    hits = store.query([1.0, 1.0, 0.5], top_k=1)
    hit_id = hits[0].id

    expanded = expand_window(store, hits, window=1)

    by_id = {c.id: c for c in expanded}
    assert by_id[hit_id].distance == hits[0].distance
    assert by_id[hit_id].distance is not None


def test_an_unplaceable_hit_still_reaches_the_model(store: ChromaStore) -> None:
    """A chunk with no attachment key gets no neighbours, but is not dropped."""
    orphan = StoredChunk(id="X:1:0", text="orphaned", metadata={"page": 1})
    assert expand_window(store, [orphan], window=2) == [orphan]


def test_windowing_reports_both_counts(store: ChromaStore) -> None:
    populate_document(store, {1: ["alpha", "beta", "gamma"]})

    result = answer_question(
        "why?",
        store=store,
        embed_query=lambda _: [1.0, 1.0, 0.5],
        answer=lambda _: reply(),
        top_k=1,
        window=1,
    )

    # `top_k` still bounds the hits; the context is what grew.
    assert result.chunks_retrieved == 1
    assert result.chunks_in_context == 3


def test_windowing_puts_the_neighbour_text_in_the_prompt(store: ChromaStore) -> None:
    populate_document(store, {1: ["the preceding sentence", "the hit", "the following one"]})
    seen: list[str] = []

    def answer(messages: list[BaseMessage]) -> AIMessage:
        seen.append(str(messages[-1].content))
        return reply()

    answer_question(
        "why?",
        store=store,
        embed_query=lambda _: [1.0, 1.0, 0.5],
        answer=answer,
        top_k=1,
        window=1,
    )

    assert "the preceding sentence" in seen[0]
    assert "the following one" in seen[0]


def test_a_neighbour_on_another_page_becomes_its_own_source(store: ChromaStore) -> None:
    """Citing the page the text is really on beats attributing it to the next one."""
    populate_document(store, {1: ["page one tail"], 2: ["page two head"]})
    hits = [chunk("page two head", page=2, chunk_index=0)]

    sources, _ = collect_sources(expand_window(store, hits, window=1))

    assert [source.page for source in sources] == [1, 2]
    assert [source.locator for source in sources] == ["p. 1", "p. 2"]


def test_windowing_leaves_an_empty_retrieval_alone(store: ChromaStore) -> None:
    assert expand_window(store, [], window=2) == []


# --- streaming (PRD-CursesUI §6.1) ---------------------------------------


USAGE = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}


def chunks_of(text: str, usage: bool = True) -> list[AIMessageChunk]:
    """The reply as a provider streams it: text deltas, then a usage-only chunk."""
    pieces = [AIMessageChunk(content=text[i : i + 4]) for i in range(0, len(text), 4)]
    if usage:
        pieces.append(AIMessageChunk(content="", usage_metadata=USAGE))  # type: ignore[arg-type]
    return pieces


def streamed(
    store: ChromaStore,
    stream: Callable[[list[BaseMessage]], Iterator[AIMessageChunk]],
    tokens: list[str],
    question: str = "why?",
    **kwargs: Any,
) -> QueryResult:
    return stream_answer_question(
        question,
        store=store,
        embed_query=lambda question: [1.0, 1.0, 0.5],
        stream=stream,
        on_token=tokens.append,
        top_k=kwargs.pop("top_k", 2),
        **kwargs,
    )


def without_timing(result: QueryResult) -> dict[str, Any]:
    dumped = result.model_dump()
    del dumped["usage"]["elapsed_s"]
    return dumped


@pytest.mark.parametrize("window", [0, 1])
def test_streaming_gives_the_same_result_as_the_blocking_call(
    store: ChromaStore, window: int
) -> None:
    """The two variants share every step but the call, so they must agree exactly."""
    populate(store, ["Writers should write every day.", "Keep previous versions of files."])
    text = "Write daily [1], and keep versions [2]."
    blocking = answer_question(
        "What should writers do?",
        store=store,
        embed_query=lambda question: [1.0, 1.0, 0.5],
        answer=lambda messages: reply(text),
        top_k=2,
        window=window,
    )
    tokens: list[str] = []
    streaming = streamed(
        store,
        lambda messages: iter(chunks_of(text)),
        tokens,
        "What should writers do?",
        window=window,
    )

    assert without_timing(streaming) == without_timing(blocking)
    assert "".join(tokens) == text


def test_the_deltas_arrive_before_the_stream_ends(store: ChromaStore) -> None:
    populate(store, ["some text"])
    tokens: list[str] = []

    def stream(messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
        for chunk in chunks_of("Because [1]."):
            yield chunk
            seen_so_far.append(len(tokens))

    seen_so_far: list[int] = []
    streamed(store, stream, tokens)
    assert seen_so_far[:3] == [1, 2, 3]  # each delta was delivered as it came


def test_the_stream_is_given_the_same_prompt(store: ChromaStore) -> None:
    populate(store, ["Writers should write every day."])
    prompts: list[list[BaseMessage]] = []

    def stream(messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
        prompts.append(messages)
        return iter(chunks_of("Yes [1]."))

    streamed(store, stream, [], "What should writers do?")
    system, human = prompts[0]
    assert str(system.content) == SYSTEM_PROMPT
    assert "write every day" in str(human.content)


def test_streamed_usage_comes_from_the_final_chunk(store: ChromaStore) -> None:
    populate(store, ["some text"])
    result = streamed(store, lambda messages: iter(chunks_of("Because [1].")), [])
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (120, 30)


def test_a_stream_without_usage_reports_zeros(store: ChromaStore) -> None:
    populate(store, ["some text"])
    result = streamed(store, lambda messages: iter(chunks_of("Because [1].", usage=False)), [])
    assert result.usage.total_tokens == 0
    assert result.answer == "Because [1]."


def test_an_empty_stream_is_an_empty_answer_not_a_crash(store: ChromaStore) -> None:
    populate(store, ["some text"])
    tokens: list[str] = []
    result = streamed(store, lambda messages: iter([]), tokens)
    assert result.answer == ""
    assert tokens == []
    assert not any(source.cited for source in result.sources)


def test_streaming_with_nothing_retrieved_skips_the_model(store: ChromaStore) -> None:
    calls: list[int] = []

    def stream(messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
        calls.append(1)
        return iter(chunks_of("unused"))

    tokens: list[str] = []
    result = streamed(store, stream, tokens)
    assert calls == []
    assert result.answer == NO_CONTEXT_ANSWER
    assert tokens == [NO_CONTEXT_ANSWER]  # a deltas-only renderer still shows it


def test_a_stream_that_fails_midway_becomes_a_user_facing_error(store: ChromaStore) -> None:
    populate(store, ["some text"])
    tokens: list[str] = []

    def stream(messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
        yield AIMessageChunk(content="Partial ")
        raise RuntimeError("connection reset")

    with pytest.raises(LLMError, match="connection reset"):
        streamed(store, stream, tokens)
    assert tokens == ["Partial "]


def test_a_failing_token_sink_is_not_blamed_on_the_model(store: ChromaStore) -> None:
    populate(store, ["some text"])

    def sink(delta: str) -> None:
        raise KeyError("ui gone")

    with pytest.raises(KeyError):
        stream_answer_question(
            "why?",
            store=store,
            embed_query=lambda question: [1.0, 1.0, 0.5],
            stream=lambda messages: iter(chunks_of("Because [1].")),
            on_token=sink,
            top_k=1,
        )


def test_block_style_deltas_are_streamed_as_text(store: ChromaStore) -> None:
    populate(store, ["some text"])
    tokens: list[str] = []
    pieces = [
        AIMessageChunk(content=[{"type": "text", "text": "Because ", "index": 0}]),
        AIMessageChunk(content=[{"type": "text", "text": "[1].", "index": 0}]),
    ]
    result = streamed(store, lambda messages: iter(pieces), tokens)
    assert tokens == ["Because ", "[1]."]
    assert result.answer == "Because [1]."


def test_the_chat_client_asks_for_usage_when_streaming() -> None:
    """Without it, OpenAI-compatible streams carry no token counts at all."""
    config = Config.model_validate(packaged_defaults())
    config.global_.api_key = "sk-test"  # never sent: building the client calls nothing
    assert build_chat(config).stream_usage is True
