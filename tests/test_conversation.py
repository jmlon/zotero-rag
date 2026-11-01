"""Follow-up questions: condensing, history in the prompt, and what is searched.

A follow-up such as "and what did the second paper find?" embeds as nothing in
particular, so retrieval must search for a standalone rewrite of it while the
model still answers the question as the user asked it, conversation in view.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from zotero_rag_core.llm import LLMError
from zotero_rag_core.querying import (
    CONDENSE_PROMPT,
    HISTORY_RULE,
    SYSTEM_PROMPT,
    Condensed,
    Turn,
    answer_question,
    build_messages,
    condense_question,
    stream_answer_question,
    strip_citations,
)
from zotero_rag_core.stores.chroma import ChromaStore

HISTORY = [
    Turn(question="Which papers study ECMP?", answer="Two do: Smith [1] and Jones [2, 3]."),
]


class Chat:
    """A chat model that records each prompt and replies with fixed text."""

    def __init__(self, reply: str = "What did Jones find about ECMP?") -> None:
        self.reply = reply
        self.prompts: list[list[BaseMessage]] = []

    def __call__(self, messages: list[BaseMessage]) -> AIMessage:
        self.prompts.append(messages)
        return AIMessage(
            content=self.reply,
            usage_metadata={"input_tokens": 50, "output_tokens": 9, "total_tokens": 59},
            response_metadata={"token_usage": {"cost": 0.0001}},
        )


# --- stripping and condensing -------------------------------------------------------


def test_citation_markers_are_stripped_from_earlier_answers() -> None:
    assert strip_citations("Two do: Smith [1] and Jones [2, 3].") == "Two do: Smith and Jones."
    assert strip_citations("No markers here.") == "No markers here."


def test_the_first_question_is_not_condensed() -> None:
    """No history, no call: the first question of a conversation costs nothing extra."""
    chat = Chat()
    condensed = condense_question("Which papers study ECMP?", [], chat)
    assert condensed == Condensed(question="Which papers study ECMP?")
    assert chat.prompts == []


def test_a_follow_up_is_condensed_with_the_conversation_in_view() -> None:
    chat = Chat()
    condensed = condense_question("And what did the second one find?", HISTORY, chat)

    assert condensed.question == "What did Jones find about ECMP?"
    assert (condensed.prompt_tokens, condensed.completion_tokens) == (50, 9)
    (prompt,) = chat.prompts
    assert isinstance(prompt[0], SystemMessage) and prompt[0].content == CONDENSE_PROMPT
    sent = str(prompt[1].content)
    assert "User: Which papers study ECMP?" in sent
    assert "Assistant: Two do: Smith and Jones." in sent  # markers stripped
    assert sent.endswith("Latest question: And what did the second one find?")


@pytest.mark.parametrize("reply", ["", "   ", '""'])
def test_an_empty_rewrite_falls_back_to_the_question(reply: str) -> None:
    condensed = condense_question("And the second one?", HISTORY, Chat(reply))
    assert condensed.question == "And the second one?"


def test_quotes_around_the_rewrite_are_removed() -> None:
    condensed = condense_question("And the second one?", HISTORY, Chat('"What did Jones find?"'))
    assert condensed.question == "What did Jones find?"


def test_a_failing_condense_call_is_a_user_facing_error() -> None:
    def broken(messages: list[BaseMessage]) -> AIMessage:
        raise RuntimeError("503 from provider")

    with pytest.raises(LLMError, match="503 from provider"):
        condense_question("And the second one?", HISTORY, broken)


# --- the prompt ----------------------------------------------------------------------


def test_without_history_the_prompt_is_exactly_a_one_off_questions() -> None:
    """The CLI and `eval` never send history; their prompt must not move at all."""
    assert build_messages("Q?", "[1] passage") == [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Context passages:\n[1] passage\n\nQuestion: Q?"),
    ]
    assert build_messages("Q?", "[1] passage", ()) == build_messages("Q?", "[1] passage")


def test_history_comes_between_the_rules_and_this_turns_passages() -> None:
    messages = build_messages("And the second one?", "[1] passage", HISTORY)

    assert [type(message) for message in messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert messages[0].content == SYSTEM_PROMPT + HISTORY_RULE
    assert messages[1].content == "Which papers study ECMP?"
    assert messages[2].content == "Two do: Smith and Jones."
    assert messages[3].content == "Context passages:\n[1] passage\n\nQuestion: And the second one?"


# --- through the query pipeline ---------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    store = ChromaStore(tmp_path / "chroma")
    store.add(
        ids=["A:1:0", "B:1:0"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["Smith measures ECMP imbalance.", "Jones finds ECMP hashing collides."],
        metadatas=[
            {
                "attachment_key": key,
                "item_key": f"I{key}",
                "title": title,
                "page": 1,
                "chunk_index": 0,
            }
            for key, title in (("A", "Smith"), ("B", "Jones"))
        ],
    )
    return store


class Recorder:
    """Seams that record what retrieval and reranking were asked."""

    def __init__(self) -> None:
        self.embedded: list[str] = []
        self.reranked: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        return [0.0, 1.0, 0.0]

    def rerank(self, question: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        self.reranked.append(question)
        return [(index, 1.0) for index in range(min(top_n, len(documents)))]


def ask_follow_up(store: ChromaStore, answer: Any, recorder: Recorder, **kwargs: Any) -> Any:
    return answer_question(
        "And the second one?",
        store=store,
        embed_query=recorder.embed,
        answer=answer,
        top_k=1,
        rerank=recorder.rerank,
        history=HISTORY,
        condensed=Condensed(question="What did Jones find?", prompt_tokens=50, cost=0.0001),
        **kwargs,
    )


def test_retrieval_and_reranking_search_for_the_condensed_question(store: ChromaStore) -> None:
    recorder = Recorder()
    ask_follow_up(store, Chat("Jones finds collisions [1]."), recorder)
    assert recorder.embedded == ["What did Jones find?"]
    assert recorder.reranked == ["What did Jones find?"]


def test_the_model_answers_the_question_as_asked(store: ChromaStore) -> None:
    chat = Chat("Jones finds collisions [1].")
    result = ask_follow_up(store, chat, Recorder())

    (prompt,) = chat.prompts
    assert str(prompt[-1].content).endswith("Question: And the second one?")
    assert prompt[1].content == "Which papers study ECMP?"
    assert result.question == "And the second one?"
    assert result.condensed.question == "What did Jones find?"
    assert result.condensed.cost == 0.0001
    assert [source.title for source in result.sources if source.cited] == ["Jones"]


def test_condensing_cost_is_kept_out_of_usage(store: ChromaStore) -> None:
    """`usage` is dumped as-is by `query --json`, so its fields must not change."""
    result = ask_follow_up(store, Chat("Jones [1]."), Recorder())
    assert set(result.usage.model_dump()) == {
        "prompt_tokens",
        "completion_tokens",
        "embedding_tokens",
        "elapsed_s",
        "answer_cost",
        "rerank_cost",
        "embedding_cost",
        "cached_tokens",
        "answer_list_cost",
    }


def test_a_one_off_question_records_no_condensing(store: ChromaStore) -> None:
    recorder = Recorder()
    result = answer_question(
        "Which papers study ECMP?",
        store=store,
        embed_query=recorder.embed,
        answer=Chat("Smith [1]."),
        top_k=1,
    )
    assert result.condensed is None
    assert recorder.embedded == ["Which papers study ECMP?"]


def test_a_streamed_follow_up_matches_the_blocking_one(store: ChromaStore) -> None:
    chat = Chat("Jones finds collisions [1].")
    blocking = ask_follow_up(store, chat, Recorder())

    prompts: list[list[BaseMessage]] = []

    def stream(messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
        prompts.append(messages)
        yield AIMessageChunk(content="Jones finds collisions [1].")

    recorder = Recorder()
    streamed = stream_answer_question(
        "And the second one?",
        store=store,
        embed_query=recorder.embed,
        stream=stream,
        on_token=lambda text: None,
        top_k=1,
        rerank=recorder.rerank,
        history=HISTORY,
        condensed=Condensed(question="What did Jones find?", prompt_tokens=50, cost=0.0001),
    )
    assert prompts == chat.prompts
    assert recorder.embedded == ["What did Jones find?"]
    assert streamed.condensed == blocking.condensed
    assert streamed.sources == blocking.sources
