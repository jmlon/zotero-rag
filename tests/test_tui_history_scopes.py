"""Prompt history and folder scopes for `zotero-rag-ui` (PRD-CursesUI §4.2, §7)."""

from __future__ import annotations

import json
from pathlib import Path

from zotero_rag.snapshot import CollectionSnapshot
from zotero_rag.tui.history import PromptHistory
from zotero_rag.tui.scopes import LIBRARY_LABEL, ScopeNode, build_scope_tree
from zotero_rag.zotero_api import CollectionNode

# --- history: stepping ------------------------------------------------------


def test_up_walks_back_and_stops_at_the_oldest() -> None:
    history = PromptHistory(["one", "two"])
    assert history.previous("") == "two"
    assert history.previous("two") == "one"
    assert history.previous("one") is None


def test_down_past_the_newest_restores_the_draft() -> None:
    history = PromptHistory(["one", "two"])
    history.previous("half-typed")
    history.previous("two")
    assert history.next() == "two"
    assert history.next() == "half-typed"
    assert history.next() is None  # already at the draft


def test_an_empty_history_recalls_nothing() -> None:
    assert PromptHistory().previous("draft") is None


def test_sending_returns_the_cursor_to_a_fresh_draft() -> None:
    history = PromptHistory(["one"])
    history.previous("draft")
    history.add("one edited")
    assert history.previous("") == "one edited"


def test_consecutive_duplicates_are_kept_once() -> None:
    history = PromptHistory()
    for prompt in ("same", "same", "other", "same"):
        history.add(prompt)
    assert history.entries == ["same", "other", "same"]


# --- history: the file ------------------------------------------------------


def test_multi_line_prompts_survive_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ui" / "history"
    history = PromptHistory.load(path)
    history.add("line one\nline two")
    history.add('quotes " and \\ backslashes')
    assert PromptHistory.load(path).entries == ["line one\nline two", 'quotes " and \\ backslashes']
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_the_file_keeps_only_the_newest_entries(tmp_path: Path) -> None:
    path = tmp_path / "history"
    history = PromptHistory.load(path, limit=3)
    for number in range(5):
        history.add(f"q{number}")
    assert history.entries == ["q2", "q3", "q4"]
    assert PromptHistory.load(path, limit=3).entries == ["q2", "q3", "q4"]


def test_a_longer_file_is_trimmed_on_load(tmp_path: Path) -> None:
    path = tmp_path / "history"
    path.write_text("".join(json.dumps(f"q{n}") + "\n" for n in range(5)), encoding="utf-8")
    assert PromptHistory.load(path, limit=2).entries == ["q3", "q4"]


def test_damaged_lines_are_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "history"
    path.write_text('"good"\nnot json\n42\n""\n"also good"\n', encoding="utf-8")
    assert PromptHistory.load(path).entries == ["good", "also good"]


def test_an_unwritable_history_still_works_for_the_session(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    history = PromptHistory.load(blocker / "history")  # its parent is a file
    history.add("question")
    assert history.entries == ["question"]


# --- scopes -----------------------------------------------------------------


def tree() -> CollectionNode:
    return CollectionNode(
        key="NET",
        name="Networking",
        children=[
            CollectionNode(key="K1", name="Routing"),
            CollectionNode(
                key="K2", name="Transport", children=[CollectionNode(key="K3", name="QUIC")]
            ),
        ],
    )


def paths(node: ScopeNode) -> list[str]:
    return [node.choice.path] + [path for child in node.children for path in paths(child)]


def test_the_pinned_root_searches_everything() -> None:
    root = build_scope_tree(CollectionSnapshot(roots=[tree()]), "ignored")
    assert root.choice.path == "Networking"
    assert root.choice.keys is None


def test_each_folder_searches_its_whole_subtree() -> None:
    root = build_scope_tree(CollectionSnapshot(roots=[tree()]), None)
    routing, transport = root.children
    assert routing.choice.keys == {"K1"}
    assert transport.choice.keys == {"K2", "K3"}
    assert transport.children[0].choice.keys == {"K3"}
    assert paths(root) == [
        "Networking",
        "Networking/Routing",
        "Networking/Transport",
        "Networking/Transport/QUIC",
    ]


def test_a_whole_library_index_gets_a_library_root() -> None:
    other = CollectionNode(key="ML", name="Machine learning")
    root = build_scope_tree(CollectionSnapshot(roots=[tree(), other]), None)
    assert root.choice.path == LIBRARY_LABEL and root.choice.keys is None
    assert [child.choice.path for child in root.children] == ["Networking", "Machine learning"]
    assert root.children[0].choice.keys == {"NET", "K1", "K2", "K3"}


def test_without_a_snapshot_there_is_only_the_root() -> None:
    root = build_scope_tree(None, "AI/KG")
    assert root.choice.path == "AI/KG" and root.choice.keys is None and root.children == []
    assert build_scope_tree(None, None).choice.path == LIBRARY_LABEL
