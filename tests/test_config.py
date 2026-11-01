"""Config loading, layering, and credential resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from zotero_rag.config import (
    Config,
    IndexConfig,
    deep_merge,
    is_index_folder,
    load_config,
    packaged_defaults,
)
from zotero_rag_core.errors import ConfigError, MissingCredentialError

ZOTERO_VARS = ("ZOTERO_LIBRARY_ID", "ZOTERO_API_KEY")
OPENROUTER_VARS = ("OPENROUTER_ENDPOINT", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real environment out of these tests."""
    for name in (*ZOTERO_VARS, *OPENROUTER_VARS):
        monkeypatch.delenv(name, raising=False)


def test_packaged_defaults_match_the_prd() -> None:
    config = Config.model_validate(packaged_defaults())
    assert config.index.chunk_size == 1000
    assert config.index.chunk_overlap == 200
    # Retrieval defaults are the ones M7 measured (§11.4.1, §11.4.2); pinned here so
    # they cannot drift back to the unmeasured PoC values unnoticed.
    assert config.index.top_k == 10
    assert config.index.retrieval == "hybrid"
    assert config.index.retrieval_window == 0
    assert config.global_.zotero_library_type == "user"
    assert config.global_.defaults.vector_db == "chroma"
    assert config.global_.defaults.reranker is None
    # A fresh config has no pinned collection.
    assert not config.index.is_pinned


def test_load_config_without_local_file_uses_packaged_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    assert config.source_dir is None
    assert config.index.chunk_size == 1000
    assert not is_index_folder(tmp_path)


def test_local_config_overrides_only_the_keys_it_sets(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "index:\n  chunk_size: 300\n  top_k: 8\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.source_dir == tmp_path
    assert config.index.chunk_size == 300
    assert config.index.top_k == 8
    # Untouched keys still come from the packaged defaults.
    assert config.index.chunk_overlap == 200
    assert config.global_.defaults.llm_model == "openai/gpt-4o-mini"
    assert is_index_folder(tmp_path)


def test_model_inheritance_prefers_the_index_override() -> None:
    config = Config.model_validate(packaged_defaults())
    assert config.embedding_model == "openai/text-embedding-3-small"
    config.index.embedding_model = "mistralai/mistral-embed"
    assert config.embedding_model == "mistralai/mistral-embed"
    # The global default is untouched by the override.
    assert config.global_.defaults.embedding_model == "openai/text-embedding-3-small"


def test_pinned_collection_round_trips(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "index:\n  collection_key: TGXVGXGP\n  collection_name: writting\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.index.is_pinned
    assert config.index.collection_key == "TGXVGXGP"
    assert config.index.collection_name == "writting"


def test_deep_merge_recurses_and_treats_null_as_a_value() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"c": 99}, "e": None}
    assert deep_merge(base, override) == {"a": {"b": 1, "c": 99}, "d": 3, "e": None}
    # The inputs are not mutated.
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        IndexConfig(chunk_size=200, chunk_overlap=200)
    with pytest.raises(ValidationError):
        IndexConfig(chunk_size=100, chunk_overlap=500)


def test_nonpositive_sizes_are_rejected() -> None:
    with pytest.raises(ValidationError):
        IndexConfig(chunk_size=0)
    with pytest.raises(ValidationError):
        IndexConfig(top_k=0)


def test_malformed_yaml_is_a_config_error(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("index: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_non_mapping_yaml_is_a_config_error(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_empty_local_config_falls_back_to_defaults(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    config = load_config(tmp_path)
    assert config.index.chunk_size == 1000
    assert config.source_dir == tmp_path


def test_invalid_value_in_local_config_names_the_file(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("index:\n  chunk_overlap: 5000\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="config.yaml"):
        load_config(tmp_path)


def test_credentials_resolve_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
    monkeypatch.setenv("ZOTERO_API_KEY", "zkey")
    monkeypatch.setenv("OPENROUTER_API_KEY", "orkey")
    config = load_config(tmp_path)
    assert config.global_.zotero_library_id == "12345"
    assert config.global_.zotero_api_key == "zkey"
    config.global_.require_zotero()
    config.global_.require_openrouter()


def test_custom_env_var_names_are_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text(
        "global:\n  zotero_api_key_env: MY_ZOTERO_KEY\n", encoding="utf-8"
    )
    monkeypatch.setenv("MY_ZOTERO_KEY", "custom")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
    config = load_config(tmp_path)
    assert config.global_.zotero_api_key == "custom"
    config.global_.require_zotero()


def test_missing_zotero_credentials_name_the_variables(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    with pytest.raises(MissingCredentialError) as excinfo:
        config.global_.require_zotero()
    assert excinfo.value.var_names == ["ZOTERO_LIBRARY_ID", "ZOTERO_API_KEY"]
    assert "ZOTERO_API_KEY" in str(excinfo.value)


def test_list_collections_does_not_require_openrouter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zotero creds alone must be enough; only index/query need OpenRouter."""
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
    monkeypatch.setenv("ZOTERO_API_KEY", "zkey")
    config = load_config(tmp_path)
    config.global_.require_zotero()
    with pytest.raises(MissingCredentialError):
        config.global_.require_openrouter()


def test_endpoint_defaults_when_the_variable_is_unset(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    assert config.global_.endpoint == "https://openrouter.ai/api/v1"


def test_secrets_are_never_serialised_back_to_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved credentials must not leak when the config is written out."""
    monkeypatch.setenv("ZOTERO_API_KEY", "supersecret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "alsosecret")
    config = load_config(tmp_path)
    dumped = config.model_dump(by_alias=True)
    assert "supersecret" not in str(dumped)
    assert "alsosecret" not in str(dumped)
    assert "zotero_api_key" not in dumped["global"]


def test_packaged_preference_indexes_transcripts_first() -> None:
    # PLAN-markdown, Phase 6: equal answer quality, ~13% fewer prompt tokens.
    config = Config.model_validate(packaged_defaults())

    assert config.index.attachment_preference[0] == "markdown"
    assert config.index.attachment_preference == Config().index.attachment_preference
