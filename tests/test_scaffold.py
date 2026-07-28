"""`init` folder scaffolding.

`init` must be entirely offline and must never destroy user data, so these
tests focus on what it writes, what it refuses to touch, and the fact that it
produces a config the loader accepts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zotero_rag.config import (
    Config,
    GlobalConfig,
    load_config,
    packaged_defaults,
    packaged_defaults_text,
)
from zotero_rag.errors import ZoteroRagError
from zotero_rag.scaffold import (
    ENV_FILENAME,
    GITIGNORE_FILENAME,
    init_index_folder,
    render_env_template,
)


@pytest.fixture
def global_config() -> GlobalConfig:
    return Config.model_validate(packaged_defaults()).global_


def test_init_creates_the_three_files(tmp_path: Path, global_config: GlobalConfig) -> None:
    target = tmp_path / "myindex"
    result = init_index_folder(target, global_config)

    assert target.is_dir()
    assert sorted(result.created) == sorted(["config.yaml", ENV_FILENAME, GITIGNORE_FILENAME])
    assert result.skipped == []
    assert (target / "config.yaml").is_file()
    assert (target / ENV_FILENAME).is_file()
    assert (target / GITIGNORE_FILENAME).is_file()


def test_init_creates_nested_directories(tmp_path: Path, global_config: GlobalConfig) -> None:
    target = tmp_path / "a" / "b" / "c"
    init_index_folder(target, global_config)
    assert (target / "config.yaml").is_file()


def test_written_config_is_the_packaged_default_verbatim(
    tmp_path: Path, global_config: GlobalConfig
) -> None:
    init_index_folder(tmp_path, global_config)
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == packaged_defaults_text()


def test_written_config_loads_and_is_unpinned(tmp_path: Path, global_config: GlobalConfig) -> None:
    """The whole point of init: the folder is immediately usable by the loader."""
    init_index_folder(tmp_path, global_config)
    config = load_config(tmp_path)
    assert config.source_dir == tmp_path
    assert not config.index.is_pinned
    assert config.index.chunk_size == 1000


def test_init_refuses_an_existing_index_folder(tmp_path: Path, global_config: GlobalConfig) -> None:
    init_index_folder(tmp_path, global_config)
    with pytest.raises(ZoteroRagError, match="already an index folder"):
        init_index_folder(tmp_path, global_config)


def test_init_never_overwrites_an_existing_env(tmp_path: Path, global_config: GlobalConfig) -> None:
    """A pre-existing .env may hold real keys — losing it would be destructive."""
    secret = "ZOTERO_API_KEY=already-a-real-key\n"
    (tmp_path / ENV_FILENAME).write_text(secret, encoding="utf-8")

    result = init_index_folder(tmp_path, global_config)

    assert (tmp_path / ENV_FILENAME).read_text(encoding="utf-8") == secret
    assert ENV_FILENAME in result.skipped
    assert ENV_FILENAME not in result.created


def test_init_never_overwrites_an_existing_gitignore(
    tmp_path: Path, global_config: GlobalConfig
) -> None:
    """The folder may sit inside a repository that already has one."""
    existing = "*.log\n"
    (tmp_path / GITIGNORE_FILENAME).write_text(existing, encoding="utf-8")

    result = init_index_folder(tmp_path, global_config)

    assert (tmp_path / GITIGNORE_FILENAME).read_text(encoding="utf-8") == existing
    assert GITIGNORE_FILENAME in result.skipped


def test_gitignore_covers_secrets_and_rebuildable_state(
    tmp_path: Path, global_config: GlobalConfig
) -> None:
    init_index_folder(tmp_path, global_config)
    content = (tmp_path / GITIGNORE_FILENAME).read_text(encoding="utf-8")
    assert ".env" in content
    assert "chroma/" in content
    assert "cache/" in content


def test_env_template_lists_every_variable_with_empty_secrets(
    tmp_path: Path, global_config: GlobalConfig
) -> None:
    init_index_folder(tmp_path, global_config)
    content = (tmp_path / ENV_FILENAME).read_text(encoding="utf-8")

    assert "ZOTERO_LIBRARY_ID=\n" in content
    assert "ZOTERO_API_KEY=\n" in content
    assert "OPENROUTER_API_KEY=\n" in content
    # The endpoint is not a secret, so it is pre-filled.
    assert "OPENROUTER_ENDPOINT=https://openrouter.ai/api/v1\n" in content


def test_env_template_follows_customised_variable_names() -> None:
    custom = GlobalConfig(zotero_api_key_env="MY_ZOTERO_KEY", api_key_env="MY_OR_KEY")
    content = render_env_template(custom)
    assert "MY_ZOTERO_KEY=\n" in content
    assert "MY_OR_KEY=\n" in content
    assert "ZOTERO_API_KEY=" not in content


def test_env_template_contains_no_real_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, global_config: GlobalConfig
) -> None:
    """A resolved key in the environment must not leak into the template."""
    monkeypatch.setenv("ZOTERO_API_KEY", "supersecret")
    init_index_folder(tmp_path, global_config)
    assert "supersecret" not in (tmp_path / ENV_FILENAME).read_text(encoding="utf-8")


def test_the_env_template_carries_the_optional_cohere_variable(tmp_path: Path) -> None:
    """It must be present but blank: needed only if reranking is switched on."""
    init_index_folder(tmp_path, GlobalConfig())

    env = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "COHERE_API_KEY=" in env
    assert "OPTIONAL" in env
    # Blank, so the empty-value rule in `load_env` ignores it entirely (§5.2).
    assert "COHERE_API_KEY=\n" in env


def test_a_renamed_cohere_variable_reaches_the_template() -> None:
    """§3.2: the names come from the config, so a customised one must follow."""
    config = GlobalConfig()
    config.cohere_api_key_env = "MY_RERANK_KEY"
    assert "MY_RERANK_KEY=" in render_env_template(config)
