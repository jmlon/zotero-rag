"""Configuration models and loader.

Secrets are never stored in `config.yaml` — only the *names* of the environment
variables that hold them. Values are resolved from the environment at load time.

Layering (lowest precedence first):

1. the default `config.yaml` shipped as package data,
2. `<directory>/config.yaml` when present (in an index folder, that *is* the
   index config; in the project root it acts as the user's global defaults),
3. CLI flags, applied by the caller after loading.
"""

from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from zotero_rag.errors import ConfigError, MissingCredentialError

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.yaml"


class Defaults(BaseModel):
    """Model / backend defaults, inherited by an index unless it overrides them."""

    llm_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    vector_db: str = "chroma"
    reranker: str | None = None


class GlobalConfig(BaseModel):
    """Endpoint, credential indirection, and inheritable defaults."""

    endpoint_env: str = "OPENROUTER_ENDPOINT"
    api_key_env: str = "OPENROUTER_API_KEY"
    zotero_library_id_env: str = "ZOTERO_LIBRARY_ID"
    zotero_api_key_env: str = "ZOTERO_API_KEY"
    zotero_library_type: Literal["user", "group"] = "user"
    defaults: Defaults = Field(default_factory=Defaults)

    # Resolved from the environment by `resolve_env()`; never read from YAML.
    endpoint: str = Field(default="https://openrouter.ai/api/v1", exclude=True)
    api_key: str = Field(default="", exclude=True)
    zotero_library_id: str = Field(default="", exclude=True)
    zotero_api_key: str = Field(default="", exclude=True)

    def resolve_env(self) -> None:
        """Populate the resolved credential fields from the environment.

        Missing values are left empty rather than raising: `list-collections`
        needs only the Zotero pair, so each command asserts just what it uses
        via `require_zotero()` / `require_openrouter()`.
        """
        self.endpoint = os.environ.get(self.endpoint_env) or self.endpoint
        self.api_key = os.environ.get(self.api_key_env, "")
        self.zotero_library_id = os.environ.get(self.zotero_library_id_env, "")
        self.zotero_api_key = os.environ.get(self.zotero_api_key_env, "")

    def require_zotero(self) -> None:
        missing = [
            name
            for name, value in (
                (self.zotero_library_id_env, self.zotero_library_id),
                (self.zotero_api_key_env, self.zotero_api_key),
            )
            if not value
        ]
        if missing:
            raise MissingCredentialError(missing, "Zotero")

    def require_openrouter(self) -> None:
        if not self.api_key:
            raise MissingCredentialError([self.api_key_env], "OpenRouter")


class IndexConfig(BaseModel):
    """Per-index settings, including the collection pinned by the first `index`."""

    collection_path: str | None = None
    collection_key: str | None = None
    collection_name: str | None = None
    recursive: bool = True

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=5, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)

    # null => inherit from global.defaults
    llm_model: str | None = None
    embedding_model: str | None = None
    vector_db: str | None = None

    @model_validator(mode="after")
    def _overlap_below_size(self) -> IndexConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @property
    def is_pinned(self) -> bool:
        """True once a collection has been pinned by a first `index` run."""
        return self.collection_key is not None


class Config(BaseModel):
    """The merged configuration for a run."""

    model_config = ConfigDict(populate_by_name=True)

    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    index: IndexConfig = Field(default_factory=IndexConfig)

    # Directory the index config was loaded from, when there was one.
    source_dir: Path | None = Field(default=None, exclude=True)

    @property
    def llm_model(self) -> str:
        return self.index.llm_model or self.global_.defaults.llm_model

    @property
    def embedding_model(self) -> str:
        return self.index.embedding_model or self.global_.defaults.embedding_model

    @property
    def vector_db(self) -> str:
        return self.index.vector_db or self.global_.defaults.vector_db


def load_env() -> None:
    """Load `.env` following python-dotenv's search: cwd, upwards, then home."""
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
        logger.debug("loaded environment from %s", found)
    home_env = Path.home() / ".env"
    if home_env.is_file():
        load_dotenv(home_env, override=False)
        logger.debug("loaded environment from %s", home_env)


def packaged_defaults() -> dict[str, Any]:
    """The default config.yaml shipped as package data."""
    text = resources.files("zotero_rag.data").joinpath(CONFIG_FILENAME).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):  # pragma: no cover - packaged file is well-formed
        raise ConfigError("Packaged default config.yaml is not a mapping")
    return data


def packaged_defaults_text() -> str:
    """Raw text of the packaged default config, for `init` to copy verbatim."""
    return resources.files("zotero_rag.data").joinpath(CONFIG_FILENAME).read_text(encoding="utf-8")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`, returning a new dict.

    An explicit `null` in the override is a real value (it means "inherit") and
    replaces the base entry, so only *absent* keys fall back to the defaults.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def is_index_folder(directory: Path) -> bool:
    """An index folder is marked by the presence of config.yaml."""
    return (directory / CONFIG_FILENAME).is_file()


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def load_config(directory: Path | None = None) -> Config:
    """Load the packaged defaults, merged with `directory/config.yaml` if present.

    `directory` defaults to the current working directory. Credentials are
    resolved from the environment; commands assert what they need themselves.
    """
    directory = Path.cwd() if directory is None else directory
    layered = packaged_defaults()
    source_dir: Path | None = None

    local = directory / CONFIG_FILENAME
    if local.is_file():
        layered = deep_merge(layered, read_yaml(local))
        source_dir = directory
        logger.debug("merged config from %s", local)

    try:
        config = Config.model_validate(layered)
    except ValueError as exc:
        location = local if source_dir else "packaged defaults"
        raise ConfigError(f"Invalid configuration in {location}: {exc}") from exc

    config.source_dir = source_dir
    config.global_.resolve_env()
    return config
