"""`zotero-rag-ui` driven headless through Textual's Pilot (PRD-CursesUI §10).

The store is a real Chroma instance with folder membership; only the embedder
and the chat model are fakes, as in `test_querying`.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessage
from textual.pilot import Pilot
from textual.widgets import Input, Markdown, Static

from zotero_rag.index_folder import Notice
from zotero_rag.snapshot import CollectionSnapshot
from zotero_rag.tui import app as app_module
from zotero_rag.tui.app import UNAVAILABLE_NOTICE, ZoteroRagApp
from zotero_rag.tui.history import PromptHistory
from zotero_rag.tui.session import QuerySession, QuerySettings
from zotero_rag.tui.widgets import (
    Entry,
    ExportScreen,
    HelpScreen,
    PromptArea,
    ScopeTree,
    Transcript,
)
from zotero_rag_core.errors import ZoteroRagError
from zotero_rag_core.querying import NO_CONTEXT_ANSWER
from zotero_rag_core.vectorstore import VectorStore
from zotero_rag_core.zotero_api import CollectionNode

SIZE = (120, 40)

#: attachment → (collection keys, text). Routing (K1) and Transport (K2 ⊃ QUIC K3).
LIBRARY = {
    "A": (["K1"], "BGP route flap damping suppresses unstable routes."),
    "B": (["K2"], "TCP congestion control backs off on loss."),
    "C": (["K3"], "QUIC avoids head-of-line blocking across streams."),
}

SNAPSHOT = CollectionSnapshot(
    roots=[
        CollectionNode(
            key="NET",
            name="Networking",
            children=[
                CollectionNode(key="K1", name="Routing"),
                CollectionNode(
                    key="K2", name="Transport", children=[CollectionNode(key="K3", name="QUIC")]
                ),
            ],
        )
    ]
)


def populate(store: VectorStore) -> None:
    store.add(
        ids=[f"{key}:1:0" for key in LIBRARY],
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
        documents=[text for _, text in LIBRARY.values()],
        metadatas=[
            {
                "item_key": f"I{key}",
                "attachment_key": key,
                "title": f"Paper {key}",
                "page": 1,
                "chunk_index": 0,
                "zotero_url": f"https://www.zotero.org/u/items/I{key}",
                "collection_keys": keys,
            }
            for key, (keys, _) in LIBRARY.items()
        ],
    )


class Model:
    """Streams a fixed reply in small pieces and records each prompt."""

    def __init__(self, reply: str = "It suppresses **unstable** routes [1].") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
        self.prompts.append(str(messages[-1].content))
        for start in range(0, len(self.reply), 5):
            yield AIMessageChunk(content=self.reply[start : start + 5])


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    store = VectorStore(tmp_path / "chroma")
    populate(store)
    return store


def session(
    store: VectorStore,
    model: Callable[[list[BaseMessage]], Iterator[AIMessageChunk]],
    snapshot: CollectionSnapshot | None = SNAPSHOT,
    **kwargs: Any,
) -> QuerySession:
    settings = QuerySettings(
        top_k=3, window=0, mode="vector", reranker=None, rerank_fanout=4, model="test/model"
    )
    return QuerySession(
        store=store,
        settings=settings,
        embed_query=lambda question: [1.0, 0.0, 0.0],
        stream=model,
        snapshot=snapshot,
        root_name="Networking",
        **kwargs,
    )


def run(
    app: ZoteroRagApp,
    steps: Callable[[Pilot[None]], Awaitable[object]],
    size: tuple[int, int] = SIZE,
) -> None:
    async def drive() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await steps(pilot)

    asyncio.run(drive())


async def ask(pilot: Pilot[None], question: str) -> Entry:
    """Type a question, send it, and wait for the answer to be complete."""
    app = pilot.app
    assert isinstance(app, ZoteroRagApp)
    app.query_one(PromptArea).focus()
    await pilot.press(*question)
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()
    return list(app.query(Entry))[-1]


def text_of(widget: Static) -> str:
    return str(widget.render())


def answer_of(entry: Entry) -> str:
    return entry.query_one(Markdown).source


def status(app: ZoteroRagApp) -> str:
    return text_of(app.query_one("#status", Static))


def select(app: ZoteroRagApp, path: str) -> None:
    tree = app.query_one(ScopeTree)
    node = tree.root
    for name in path.split("/")[1:]:
        node = next(
            child for child in node.children if child.data and child.data.path.endswith(name)
        )
    tree.select_node(node)


# --- launch ---------------------------------------------------------------


def test_launch_selects_the_root_and_focuses_the_prompt(store: VectorStore) -> None:
    app = ZoteroRagApp(session(store, Model()), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        tree = app.query_one(ScopeTree)
        assert tree.active is tree.root and tree.root.is_expanded
        assert [str(child.label) for child in tree.root.children] == ["Routing", "Transport"]
        assert isinstance(app.focused, PromptArea)
        assert "Scope: Networking" in status(app) and "top_k 3" in status(app)

    run(app, steps)


def test_tab_cycles_prompt_tree_results(store: VectorStore) -> None:
    app = ZoteroRagApp(session(store, Model()), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        seen = []
        for _ in range(3):
            await pilot.press("tab")
            seen.append(type(app.focused).__name__)
        assert seen == ["ScopeTree", "Transcript", "PromptArea"]

    run(app, steps)


# --- asking -----------------------------------------------------------------


def test_an_answer_streams_in_with_its_cited_sources(store: VectorStore) -> None:
    model = Model()
    app = ZoteroRagApp(session(store, model), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        entry = await ask(pilot, "What is flap damping?")
        assert entry.scope_path == "Networking"
        assert "[Networking]" in text_of(entry.query_one(".scope-tag", Static))
        assert "> What is flap damping?" in text_of(entry.query_one(".question", Static))
        assert answer_of(entry) == model.reply
        footnote = text_of(entry.query_one(".footnote", Static))
        assert "Sources" in footnote and "[1] Paper A" in footnote
        assert "https://www.zotero.org/u/items/IA" in footnote
        assert app.query_one(PromptArea).text == ""
        assert "retrieving" not in status(app) and not app.busy

    run(app, steps)


def test_a_selected_folder_limits_the_search_to_its_subtree(store: VectorStore) -> None:
    model = Model("From the evidence [1].")
    app = ZoteroRagApp(session(store, model), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        select(app, "Networking/Transport")
        await pilot.pause()
        assert "Scope: Networking/Transport" in status(app)
        entry = await ask(pilot, "congestion?")
        assert "[Networking/Transport]" in text_of(entry.query_one(".scope-tag", Static))
        assert "Paper B" in model.prompts[-1] and "Paper C" in model.prompts[-1]
        assert "Paper A" not in model.prompts[-1]

    run(app, steps)


def test_moving_the_cursor_does_not_change_the_scope(store: VectorStore) -> None:
    app = ZoteroRagApp(session(store, Model()), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        app.query_one(ScopeTree).focus()
        await pilot.press("down", "down")
        assert "Scope: Networking ·" in status(app)
        await pilot.press("enter")
        assert "Scope: Networking/Transport" in status(app)

    run(app, steps)


def test_an_empty_prompt_is_not_sent(store: VectorStore) -> None:
    model = Model()
    app = ZoteroRagApp(session(store, model), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        await pilot.press("space", "enter")
        await pilot.pause()
        assert list(app.query(Entry)) == [] and model.prompts == []

    run(app, steps)


def test_alt_enter_starts_a_new_line_and_question_mark_is_typed(store: VectorStore) -> None:
    app = ZoteroRagApp(session(store, Model()), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        await pilot.press("a", "alt+enter", "b", "question_mark")
        assert app.query_one(PromptArea).text == "a\nb?"
        assert not isinstance(app.screen, HelpScreen)

    run(app, steps)


def test_a_failing_model_shows_an_error_and_the_session_goes_on(store: VectorStore) -> None:
    def failing(messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
        yield AIMessageChunk(content="Partial")
        raise RuntimeError("402 insufficient credits")

    app = ZoteroRagApp(session(store, failing), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        entry = await ask(pilot, "why?")
        assert answer_of(entry) == "Partial"
        footnote = entry.query_one(".footnote", Static)
        assert footnote.has_class("error") and "402 insufficient credits" in text_of(footnote)
        assert not app.busy

    run(app, steps)


def test_a_folder_with_nothing_in_it_says_it_may_be_too_narrow(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    populate(store)
    empty = SNAPSHOT.model_copy(deep=True)
    empty.roots[0].children.append(CollectionNode(key="EMPTY", name="Empty"))
    model = Model()
    app = ZoteroRagApp(session(store, model, snapshot=empty), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        select(app, "Networking/Empty")
        entry = await ask(pilot, "anything?")
        assert answer_of(entry) == NO_CONTEXT_ANSWER
        assert "too narrow" in text_of(entry.query_one(".footnote", Static))
        assert model.prompts == []

    run(app, steps)


# --- history ----------------------------------------------------------------


def test_up_recalls_the_previous_prompt_and_the_history_is_saved(
    store: VectorStore, tmp_path: Path
) -> None:
    path = tmp_path / ".zotero-rag-ui" / "history"
    app = ZoteroRagApp(session(store, Model()), PromptHistory.load(path))

    async def steps(pilot: Pilot[None]) -> None:
        await ask(pilot, "first")
        await pilot.press("d", "r", "a", "f", "t", "up")
        assert app.query_one(PromptArea).text == "first"
        await pilot.press("down")
        assert app.query_one(PromptArea).text == "draft"

    run(app, steps)
    assert PromptHistory.load(path).entries == ["first"]


# --- folder filtering unavailable --------------------------------------------


def test_without_a_snapshot_the_tree_is_the_root_and_questions_still_work(
    store: VectorStore,
) -> None:
    app = ZoteroRagApp(session(store, Model(), snapshot=None), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        tree = app.query_one(ScopeTree)
        assert not tree.root.children and not tree.root.allow_expand
        notes = [text_of(note) for note in app.query(".notice").results(Static)]
        assert notes == [UNAVAILABLE_NOTICE]
        entry = await ask(pilot, "flap damping?")
        assert "[1] Paper A" in text_of(entry.query_one(".footnote", Static))

    run(app, steps)


def test_open_index_notices_appear_in_the_transcript(store: VectorStore) -> None:
    notice = Notice("warning", "chunk_size changed since indexing.")
    app = ZoteroRagApp(session(store, Model(), notices=[notice]), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        notes = [text_of(note) for note in app.query(".notice").results(Static)]
        assert notes == ["Warning: chunk_size changed since indexing."]

    run(app, steps)


# --- screen -----------------------------------------------------------------


def test_help_opens_with_f1_and_closes_with_escape(store: VectorStore) -> None:
    app = ZoteroRagApp(session(store, Model()), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        await pilot.press("f1")
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("escape")
        assert not isinstance(app.screen, HelpScreen)

    run(app, steps)


def test_a_small_terminal_shows_a_message_instead_of_the_layout(store: VectorStore) -> None:
    app = ZoteroRagApp(session(store, Model()), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        assert app.query_one("#too-small").display
        assert not app.query_one("#main").display

    run(app, steps, size=(70, 20))


def test_a_keyword_index_failure_is_reported_on_the_question(store: VectorStore) -> None:
    hybrid = session(store, Model())
    hybrid.settings = QuerySettings(
        top_k=3, window=0, mode="hybrid", reranker=None, rerank_fanout=4, model="m"
    )
    hybrid._keyword_ready.clear()
    hybrid._keyword_error = ZoteroRagError("disk full")
    hybrid._keyword_ready.set()
    app = ZoteroRagApp(hybrid, PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        entry = await ask(pilot, "why?")
        assert "keyword index could not be built" in text_of(entry.query_one(".footnote", Static))

    run(app, steps)


def test_transcript_follows_new_output(store: VectorStore) -> None:
    app = ZoteroRagApp(session(store, Model("line\n\n" * 40 + "[1]")), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        await ask(pilot, "long?")
        transcript = app.query_one(Transcript)
        await pilot.pause()
        assert transcript.max_scroll_y > 0
        assert transcript.scroll_y == transcript.max_scroll_y

    run(app, steps)


def test_the_answer_is_drawn_while_it_streams_and_completed_at_the_end(
    store: VectorStore,
) -> None:
    """The model pauses mid-answer until the partial text is on screen."""
    release = threading.Event()

    def paused(messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
        yield AIMessageChunk(content="Partial ")
        assert release.wait(timeout=10)
        yield AIMessageChunk(content="rest [1].")

    app = ZoteroRagApp(session(store, paused), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        await pilot.press(*"why?", "enter")
        entry = app.query_one(Entry)
        for _ in range(100):
            if answer_of(entry):
                break
            await pilot.pause(0.05)
        assert answer_of(entry) == "Partial "
        assert "answering" in status(app)
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert answer_of(entry) == "Partial rest [1]."
        assert "[1] Paper A" in text_of(entry.query_one(".footnote", Static))

    run(app, steps)


def test_history_survives_a_restart_and_a_recalled_prompt_can_be_edited(
    store: VectorStore, tmp_path: Path
) -> None:
    """Criterion 5: recall with ↑, edit, resend; all of it still there next launch."""
    path = tmp_path / ".zotero-rag-ui" / "history"
    run(ZoteroRagApp(session(store, Model()), PromptHistory.load(path)), lambda p: ask(p, "first"))

    second = ZoteroRagApp(session(store, Model()), PromptHistory.load(path))

    async def steps(pilot: Pilot[None]) -> None:
        await pilot.press("up")
        assert second.query_one(PromptArea).text == "first"
        entry = await ask(pilot, " again")
        assert entry.question == "first again"

    run(second, steps)
    assert PromptHistory.load(path).entries == ["first", "first again"]


# --- export and copy --------------------------------------------------------


def test_ctrl_l_copies_the_latest_answer_with_its_sources(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = ZoteroRagApp(session(store, Model()), PromptHistory())
    copied: list[str] = []
    monkeypatch.setattr(app_module, "copy_with_system_tool", lambda text: copied.append(text))
    sent: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", sent.append)

    async def steps(pilot: Pilot[None]) -> None:
        await pilot.press("ctrl+l")
        assert copied == [] and sent == []  # nothing to copy yet
        await ask(pilot, "first?")
        model_two = Model("Second answer [1].")
        app.session._stream = model_two
        await ask(pilot, "second?")
        await pilot.press("ctrl+l")

    run(app, steps)
    assert copied == sent
    (text,) = copied
    assert text.startswith("Second answer [1].\n\n**Sources**")
    assert "Paper A" in text


def test_ctrl_s_saves_the_transcript_where_the_user_says(
    store: VectorStore, tmp_path: Path
) -> None:
    app = ZoteroRagApp(session(store, Model(), index_dir=tmp_path), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        await ask(pilot, "What is flap damping?")
        await pilot.press("ctrl+s")
        assert isinstance(app.screen, ExportScreen)
        field = app.screen.query_one(Input)
        assert field.value.startswith("transcripts/zotero-rag-") and field.value.endswith(".md")
        field.value = "notes/session.md"
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, ExportScreen)

    run(app, steps)
    text = (tmp_path / "notes" / "session.md").read_text(encoding="utf-8")
    assert text.startswith("# zotero-rag transcript\n\n- Collection: Networking\n")
    assert "> What is flap damping?" in text and "Paper A" in text


def test_an_existing_file_is_replaced_only_on_a_second_enter(
    store: VectorStore, tmp_path: Path
) -> None:
    existing = tmp_path / "keep.md"
    existing.write_text("precious", encoding="utf-8")
    app = ZoteroRagApp(session(store, Model(), index_dir=tmp_path), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        await ask(pilot, "q?")
        await pilot.press("ctrl+s")
        assert isinstance(app.screen, ExportScreen)
        app.screen.query_one(Input).value = "keep.md"
        await pilot.press("enter")
        assert isinstance(app.screen, ExportScreen)
        assert "already exists" in text_of(app.screen.query_one("#export-status", Static))
        assert existing.read_text(encoding="utf-8") == "precious"
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, ExportScreen)

    run(app, steps)
    assert existing.read_text(encoding="utf-8").startswith("# zotero-rag transcript")


def test_an_unwritable_path_is_reported_in_the_dialog(store: VectorStore, tmp_path: Path) -> None:
    (tmp_path / "file").write_text("", encoding="utf-8")
    app = ZoteroRagApp(session(store, Model(), index_dir=tmp_path), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        await ask(pilot, "q?")
        await pilot.press("ctrl+s")
        assert isinstance(app.screen, ExportScreen)
        app.screen.query_one(Input).value = "file/under/a/file.md"
        await pilot.press("enter")
        assert isinstance(app.screen, ExportScreen)
        assert "Cannot save" in text_of(app.screen.query_one("#export-status", Static))
        await pilot.press("escape")
        assert not isinstance(app.screen, ExportScreen)

    run(app, steps)


def test_nothing_to_save_before_the_first_question(store: VectorStore) -> None:
    app = ZoteroRagApp(session(store, Model()), PromptHistory())

    async def steps(pilot: Pilot[None]) -> None:
        await pilot.press("ctrl+s")
        assert not isinstance(app.screen, ExportScreen)

    run(app, steps)
