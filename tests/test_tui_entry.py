"""`zotero-rag-ui` start-up and its query session (PRD-CursesUI §2, §5, §9, §12).

Nothing here draws a screen: the app's `run` is replaced, so these check what
happens before it, and what a session answers, with the network seams faked.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from typer.testing import CliRunner

from zotero_rag import PACKAGE_LOGGERS, cli
from zotero_rag.config import Config
from zotero_rag.snapshot import write_snapshot
from zotero_rag.tui import UI_DIRNAME, main, session
from zotero_rag.tui.app import ZoteroRagApp
from zotero_rag.tui.history import HISTORY_FILENAME
from zotero_rag.tui.session import Phase, QuerySession, QuerySettings, open_session
from zotero_rag_core.fingerprint import StoreFingerprint
from zotero_rag_core.querying import answer_question
from zotero_rag_core.retrieval import KeywordIndex, RetrievalMode
from zotero_rag_core.vectorstore import VectorStore
from zotero_rag_core.zotero_api import CollectionNode

runner = CliRunner()

REPLY = "Fog nodes sit between the edge and the cloud [1]."


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No real `.env`, no real keys, and the root logger put back afterwards."""
    for module in (cli, main):
        monkeypatch.setattr(module, "load_env", lambda: None)
    for name in ("ZOTERO_LIBRARY_ID", "ZOTERO_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    package_levels = {name: logging.getLogger(name).level for name in PACKAGE_LOGGERS}
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        if handler not in handlers:
            handler.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)
    for name, package_level in package_levels.items():
        logging.getLogger(name).setLevel(package_level)


class Embeddings:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.5]


class Chat:
    """Answers both ways, so the CLI and the UI can be compared on one reply."""

    def invoke(self, messages: object) -> AIMessage:
        return AIMessage(content=REPLY)

    def stream(self, messages: object) -> Iterator[AIMessageChunk]:
        yield from (AIMessageChunk(content=word + " ") for word in REPLY.split())


@pytest.fixture
def fake_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (cli, session):
        monkeypatch.setattr(module, "build_embeddings", lambda config: Embeddings())
        monkeypatch.setattr(module, "build_chat", lambda config: Chat())


def index_folder(directory: Path, fingerprint: bool = True) -> VectorStore:
    """A folder indexed from a pinned `Fog` collection with a `Edge` sub-folder."""
    config = Config()
    store = VectorStore(directory / "chroma")
    store.add(
        ids=["A:1:0", "B:1:0"],
        embeddings=[[1.0, 0.0, 0.5], [0.9, 0.1, 0.5]],
        documents=["A fog node sits between edge devices and the cloud.", "Edge caching."],
        metadatas=[
            {
                "attachment_key": key,
                "item_key": f"I{key}",
                "title": f"Paper {key}",
                "page": 1,
                "chunk_index": 0,
                "embedding_model": config.embedding_model,
                "collection_keys": keys,
            }
            for key, keys in (("A", ["FOG"]), ("B", ["EDGE"]))
        ],
    )
    if fingerprint:
        store.write_fingerprint(StoreFingerprint.intended(config, embedding_dim=3))
    (directory / "config.yaml").write_text(
        "index:\n  top_k: 3\n  collection_name: Fog\n  collection_key: FOG\n", encoding="utf-8"
    )
    return store


def write_tree(directory: Path) -> None:
    write_snapshot(
        directory,
        [CollectionNode(key="FOG", name="Fog", children=[CollectionNode(key="EDGE", name="Edge")])],
    )


# --- refusing to start (criterion 7) ------------------------------------------


def outside(directory: Path) -> None:
    pass


def never_indexed(directory: Path) -> None:
    (directory / "config.yaml").write_text("index:\n  top_k: 3\n", encoding="utf-8")


def wrong_model(directory: Path) -> None:
    index_folder(directory)
    (directory / "config.yaml").write_text(
        "index:\n  embedding_model: new/model\n", encoding="utf-8"
    )


def no_key(directory: Path) -> None:
    index_folder(directory)


@pytest.mark.parametrize("folder", [outside, never_indexed, wrong_model, no_key])
def test_it_refuses_exactly_as_query_does_and_draws_nothing(
    folder: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder(tmp_path)
    monkeypatch.chdir(tmp_path)

    def must_not_draw(self: ZoteroRagApp) -> None:
        raise AssertionError("the screen was drawn")

    monkeypatch.setattr(ZoteroRagApp, "run", must_not_draw)

    query = runner.invoke(cli.app, ["query", "anything"])
    ui = runner.invoke(main.app, [])

    assert ui.exit_code == query.exit_code == 1
    assert ui.output == query.output
    assert "Error:" in ui.output


# --- opening a session -----------------------------------------------------


@pytest.mark.usefixtures("fake_clients")
def test_the_session_takes_its_settings_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_folder(tmp_path)
    write_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    opened = open_session(None, None, None, None)

    assert opened.settings.top_k == 3
    assert opened.settings.mode == Config().index.retrieval
    assert opened.root_name == "Fog"
    assert opened.index_dir == tmp_path
    assert opened.snapshot is not None and opened.snapshot.all_keys() == {"FOG", "EDGE"}
    assert opened.notices == []


@pytest.mark.usefixtures("fake_clients")
def test_launch_flags_override_the_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    opened = open_session(7, 2, "hybrid", "none")

    assert opened.settings == QuerySettings(
        top_k=7,
        window=2,
        mode="hybrid",
        reranker=None,
        rerank_fanout=Config().index.reranker_fanout,
        model=Config().llm_model,
    )
    assert opened.snapshot is None  # indexed before folders were recorded
    assert not opened.keyword_index_ready  # built after the screen is up


@pytest.mark.usefixtures("fake_clients")
def test_setup_notices_are_kept_for_the_transcript_not_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index_folder(tmp_path, fingerprint=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    opened = open_session(None, None, None, None)

    assert [notice.level for notice in opened.notices] == ["note"]
    assert "predates fingerprinting" in opened.notices[0].text
    assert capsys.readouterr().err == ""


# --- launching ----------------------------------------------------------------


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[ZoteroRagApp]:
    apps: list[ZoteroRagApp] = []
    monkeypatch.setattr(ZoteroRagApp, "run", lambda self: apps.append(self))
    return apps


@pytest.mark.usefixtures("fake_clients")
def test_it_launches_with_the_folder_history_and_no_terminal_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launched: list[ZoteroRagApp]
) -> None:
    index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    result = runner.invoke(main.app, [])

    assert result.exit_code == 0, result.output
    (app,) = launched
    assert app.history.path == tmp_path / UI_DIRNAME / HISTORY_FILENAME
    # Python's last-resort handler would write warnings over the screen.
    assert [type(handler) for handler in logging.getLogger().handlers] == [logging.NullHandler]
    assert not (tmp_path / UI_DIRNAME).exists()  # nothing is written until a prompt is sent


@pytest.mark.usefixtures("fake_clients")
def test_verbose_logs_to_a_file_in_the_index_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launched: list[ZoteroRagApp]
) -> None:
    index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    assert runner.invoke(main.app, ["-v"]).exit_code == 0

    (handler,) = logging.getLogger().handlers
    assert isinstance(handler, logging.FileHandler)
    assert Path(handler.baseFilename) == tmp_path / UI_DIRNAME / main.LOG_FILENAME
    logging.getLogger("zotero_rag.test").info("hello from the test")
    handler.flush()
    log = (tmp_path / UI_DIRNAME / main.LOG_FILENAME).read_text(encoding="utf-8")
    assert "hello from the test" in log
    assert "sk-test" not in log


def test_version_and_help() -> None:
    assert "zotero-rag-ui" in runner.invoke(main.app, ["--version"]).output
    assert "--retrieval" in runner.invoke(main.app, ["--help"]).output


# --- what a session answers ---------------------------------------------------


def make_session(
    store: VectorStore, mode: RetrievalMode = "vector", window: int = 0
) -> QuerySession:
    return QuerySession(
        store=store,
        settings=QuerySettings(
            top_k=2, window=window, mode=mode, reranker=None, rerank_fanout=4, model="m"
        ),
        embed_query=Embeddings().embed_query,
        stream=Chat().stream,
    )


def ask(asked: QuerySession, keys: frozenset[str] | None) -> Any:
    return asked.ask(
        "what is a fog node?", keys, on_token=lambda delta: None, on_phase=lambda phase: None
    )


@pytest.mark.parametrize(("mode", "window"), [("vector", 0), ("hybrid", 1)])
def test_the_root_scope_gives_the_same_sources_as_query(
    tmp_path: Path, mode: RetrievalMode, window: int
) -> None:
    """Criterion 2: the root is the unscoped search `zotero-rag query` makes."""
    store = index_folder(tmp_path)
    asked = make_session(store, mode, window)
    asked.build_keyword_index()

    from_ui = ask(asked, None)
    from_query = answer_question(
        "what is a fog node?",
        store=store,
        embed_query=Embeddings().embed_query,
        answer=lambda messages: Chat().invoke(messages),
        top_k=2,
        window=window,
        mode=mode,
        keyword_index=KeywordIndex.from_store(store) if mode == "hybrid" else None,
    )

    assert from_ui.sources == from_query.sources
    assert from_ui.answer == from_query.answer


def test_a_sub_folder_cites_only_its_own_items(tmp_path: Path) -> None:
    """Criterion 3: every source of a scoped answer belongs to the subtree."""
    asked = make_session(index_folder(tmp_path))
    result = ask(asked, frozenset({"EDGE"}))
    assert {source.attachment_key for source in result.sources} == {"B"}


def test_a_question_waits_for_the_keyword_index(tmp_path: Path) -> None:
    """§9: a question sent while BM25 is still building is answered once it is ready."""
    asked = make_session(index_folder(tmp_path), mode="hybrid")
    phases: list[Phase] = []
    results: list[Any] = []

    def question() -> None:
        results.append(asked.ask("fog?", None, on_token=lambda delta: None, on_phase=phases.append))

    worker = threading.Thread(target=question)
    worker.start()
    worker.join(timeout=0.3)
    assert worker.is_alive() and phases == ["waiting"]

    asked.build_keyword_index()
    worker.join(timeout=10)
    assert phases == ["waiting", "retrieving", "answering"]
    assert results[0].answer == REPLY.strip()


def test_phases_of_an_ordinary_question(tmp_path: Path) -> None:
    asked = make_session(index_folder(tmp_path))
    phases: list[Phase] = []

    def stream(messages: list[BaseMessage]) -> Iterator[AIMessageChunk]:
        assert phases == ["retrieving", "answering"]  # retrieval is over before the call
        yield AIMessageChunk(content="ok")

    asked._stream = stream
    asked.ask("fog?", None, on_token=lambda delta: None, on_phase=phases.append)
    assert phases == ["retrieving", "answering"]
