"""How `.env` files are found and layered.

The case that matters: `init` writes a `.env` *template* with empty values into
the index folder, so a first-match-wins search would let that template shadow the
real keys a user keeps one directory up.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zotero_rag.config import ENV_FILENAME, env_file_candidates, existing_env_files, load_env

CREDENTIALS = ("ZOTERO_LIBRARY_ID", "ZOTERO_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_ENDPOINT")


@pytest.fixture(autouse=True)
def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CREDENTIALS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point `Path.home()` at a scratch directory, so ~/.env cannot leak in."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    return fake_home


def write_env(directory: Path, **values: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ENV_FILENAME
    body = "".join(f"{name}={value}\n" for name, value in values.items())
    path.write_text(body, encoding="utf-8")
    return path


def test_candidates_run_from_the_folder_upward_then_home(home: Path, tmp_path: Path) -> None:
    candidates = env_file_candidates(tmp_path / "parent" / "index")
    assert candidates[0] == tmp_path / "parent" / "index" / ENV_FILENAME
    assert candidates[1] == tmp_path / "parent" / ENV_FILENAME
    assert candidates[-1] == home / ENV_FILENAME


def test_an_empty_template_does_not_shadow_the_parent(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported bug: `init`'s blank .env hid the real keys one level up."""
    write_env(tmp_path, OPENROUTER_API_KEY="parent-key", ZOTERO_API_KEY="parent-zotero")
    index_dir = tmp_path / "Networking"
    write_env(index_dir, ZOTERO_LIBRARY_ID="", ZOTERO_API_KEY="", OPENROUTER_API_KEY="")
    monkeypatch.chdir(index_dir)

    load_env()

    assert os.environ["OPENROUTER_API_KEY"] == "parent-key"
    assert os.environ["ZOTERO_API_KEY"] == "parent-zotero"


def test_the_nearest_filled_in_value_wins(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_env(tmp_path, OPENROUTER_API_KEY="parent-key")
    index_dir = tmp_path / "index"
    write_env(index_dir, OPENROUTER_API_KEY="local-key")
    monkeypatch.chdir(index_dir)

    load_env()

    assert os.environ["OPENROUTER_API_KEY"] == "local-key"


def test_the_real_environment_beats_every_file(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_env(tmp_path, OPENROUTER_API_KEY="file-key")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "exported-key")

    load_env()

    assert os.environ["OPENROUTER_API_KEY"] == "exported-key"


def test_home_env_covers_a_folder_with_no_env_at_all(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_env(home, ZOTERO_API_KEY="home-key")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    load_env()

    assert os.environ["ZOTERO_API_KEY"] == "home-key"


def test_several_files_contribute_different_variables(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user may keep Zotero keys globally and an OpenRouter key per project."""
    write_env(home, ZOTERO_LIBRARY_ID="12345", ZOTERO_API_KEY="home-zotero")
    project = tmp_path / "project"
    write_env(project, OPENROUTER_API_KEY="project-key")
    index_dir = project / "index"
    write_env(index_dir, ZOTERO_LIBRARY_ID="", ZOTERO_API_KEY="", OPENROUTER_API_KEY="")
    monkeypatch.chdir(index_dir)

    load_env()

    assert os.environ["ZOTERO_LIBRARY_ID"] == "12345"
    assert os.environ["ZOTERO_API_KEY"] == "home-zotero"
    assert os.environ["OPENROUTER_API_KEY"] == "project-key"


def test_existing_env_files_lists_only_what_is_there(home: Path, tmp_path: Path) -> None:
    write_env(tmp_path)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    found = existing_env_files(index_dir)
    assert tmp_path / ENV_FILENAME in found
    assert index_dir / ENV_FILENAME not in found


def test_loading_with_no_env_files_anywhere_is_a_no_op(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    load_env()
    assert "OPENROUTER_API_KEY" not in os.environ
