"""CLI surface: help text, version, and clean failure without credentials.

These never reach the network: `load_env` is neutralised so a developer's real
`.env` cannot leak in, and the credential check fails before any API call.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from zotero_rag import __version__, cli
from zotero_rag.zotero_api import CollectionNode

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_env", lambda: None)
    for name in ("ZOTERO_LIBRARY_ID", "ZOTERO_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # Rich wraps to the terminal width; widen it so assertions on multi-word
    # messages are not broken by a line break landing mid-phrase.
    monkeypatch.setenv("COLUMNS", "200")


def test_help_lists_every_planned_command() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("list-collections", "init", "index", "query"):
        assert command in result.output


def test_version_flag() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(cli.app, [])
    assert result.exit_code != 0
    assert "list-collections" in result.output


def test_list_collections_without_credentials_fails_cleanly() -> None:
    result = runner.invoke(cli.app, ["list-collections"])
    assert result.exit_code == 1
    assert "ZOTERO_API_KEY" in result.output
    # A user-facing error, not a traceback.
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_unimplemented_commands_say_so_and_exit_nonzero() -> None:
    for command in ("index", "query"):
        result = runner.invoke(cli.app, [command])
        assert result.exit_code == 1, command
        assert "not implemented" in result.output, command


def test_init_works_offline_without_any_credentials(tmp_path: Path) -> None:
    """Acceptance criterion 2: init contacts nothing, so no keys are needed."""
    target = tmp_path / "idx"
    result = runner.invoke(cli.app, ["init", str(target)])
    assert result.exit_code == 0, result.output
    assert (target / "config.yaml").is_file()
    assert (target / ".env").is_file()
    assert "created" in result.output


def test_init_defaults_to_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["init"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.yaml").is_file()


def test_init_on_an_existing_index_fails_cleanly(tmp_path: Path) -> None:
    assert runner.invoke(cli.app, ["init", str(tmp_path)]).exit_code == 0
    result = runner.invoke(cli.app, ["init", str(tmp_path)])
    assert result.exit_code == 1
    assert "already an index folder" in result.output


def test_depth_option_rejects_zero() -> None:
    result = runner.invoke(cli.app, ["list-collections", "--depth", "0"])
    assert result.exit_code != 0


def _render(nodes: list[CollectionNode], depth: int | None = None) -> str:
    out = StringIO()
    Console(file=out, width=200, no_color=True, highlight=False).print(
        cli._render_tree(nodes, depth)
    )
    return out.getvalue()


def test_collection_names_are_not_parsed_as_rich_markup() -> None:
    """A name containing square brackets must render literally, not as a tag."""
    text = _render([CollectionNode(key="K1", name="[Draft] Notes", num_items=2)])
    assert "[Draft] Notes" in text
    assert "K1" in text


def test_item_counts_are_pluralised() -> None:
    text = _render(
        [
            CollectionNode(key="ONE", name="Single", num_items=1),
            CollectionNode(key="MANY", name="Several", num_items=6),
        ]
    )
    assert "1 item" in text
    assert "1 items" not in text
    assert "6 items" in text


def test_depth_limit_hides_deeper_levels_and_says_how_many() -> None:
    tree = [
        CollectionNode(
            key="ROOT",
            name="Research",
            num_items=1,
            children=[
                CollectionNode(key="A", name="Alpha"),
                CollectionNode(key="B", name="Beta"),
            ],
        )
    ]
    full = _render(tree)
    assert "Alpha" in full and "Beta" in full

    shallow = _render(tree, depth=1)
    assert "Research" in shallow
    assert "Alpha" not in shallow
    assert "+2 subcollections" in shallow
