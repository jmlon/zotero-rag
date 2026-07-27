"""The query pipeline: retrieval, citation numbering, prompt, and usage.

The store is a real on-disk Chroma instance; only the two network seams
(embedding the question, calling the chat model) are local fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from zotero_rag.llm import LLMError, message_text, usage_from_message
from zotero_rag.querying import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    answer_question,
    build_context,
    cited_numbers,
    collect_sources,
)
from zotero_rag.vectorstore import StoredChunk, VectorStore

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
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma")


def populate(store: VectorStore, texts: list[str]) -> None:
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


def test_answers_from_the_retrieved_chunks(store: VectorStore) -> None:
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


def test_top_k_bounds_the_retrieval(store: VectorStore) -> None:
    populate(store, [f"passage {number}" for number in range(1, 6)])
    result = answer_question(
        "anything",
        store=store,
        embed_query=lambda question: [1.0, 1.0, 0.5],
        answer=lambda messages: reply(),
        top_k=2,
    )
    assert result.chunks_retrieved == 2


def test_usage_is_reported_for_the_verbose_summary(store: VectorStore) -> None:
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
    store: VectorStore,
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


def test_a_failing_llm_call_becomes_a_user_facing_error(store: VectorStore) -> None:
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


def test_sources_record_whether_the_answer_cited_them(store: VectorStore) -> None:
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


def test_only_the_cited_sources_are_marked(store: VectorStore) -> None:
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
