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

import json
import logging
import os
import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from zotero_rag_core.config import (  # noqa: F401 - re-exported for the CLI
    COHERE_RERANKER,
    Config,
    Defaults,
    EmbeddingModelSpec,
    EmbeddingPricing,
    EvalConfig,
    GlobalConfig,
    IndexConfig,
    MarkdownConfig,
    QualityThresholds,
    RerankerName,
    RerankerPricing,
    RerankerSpec,
    is_reranker_name,
)
from zotero_rag_core.errors import ConfigError

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.yaml"
ENV_FILENAME = ".env"


def env_file_candidates(directory: Path | None = None) -> list[Path]:
    """Every `.env` path consulted, nearest first: `directory`, its ancestors, home."""
    start = Path.cwd() if directory is None else directory
    candidates = [parent / ENV_FILENAME for parent in (start, *start.parents)]
    home_env = Path.home() / ENV_FILENAME
    if home_env not in candidates:
        candidates.append(home_env)
    return candidates


def existing_env_files(directory: Path | None = None) -> list[Path]:
    """The `.env` files that actually exist, nearest first."""
    return [path for path in env_file_candidates(directory) if path.is_file()]


def load_env() -> None:
    """Load credentials from every `.env` from the current folder upward, then home.

    Unlike python-dotenv's own search, this does not stop at the nearest file, and
    it ignores entries with an empty value. Both matter because `init` writes a
    `.env` *template* into the index folder: with a first-match-wins search, that
    empty template would shadow the real keys a user keeps in the parent folder or
    at `~/.env`, and the run would fail with "missing credentials" while pointing
    at a file that looks correctly filled in.

    A variable already set in the real environment always wins, and among the
    files the nearest non-empty value wins.
    """
    for path in existing_env_files():
        applied: list[str] = []
        for name, value in dotenv_values(path).items():
            # An empty entry means "not filled in yet", so it must not shadow a
            # value that a file further up does provide.
            if value and not os.environ.get(name):
                os.environ[name] = value
                applied.append(name)
        if applied:
            # Names only — never log a credential's value.
            logger.info("loaded %s from %s", ", ".join(applied), path)


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


_PIN_KEYS = ("collection_path", "collection_key", "collection_name")
_DEFAULT_INDENT = "    "


def _yaml_scalar(value: str | None) -> str:
    """Render a value as a YAML scalar. JSON strings are valid YAML."""
    return "null" if value is None else json.dumps(value)


def _index_block_bounds(lines: list[str]) -> tuple[int, int] | None:
    """Line range (start_of_body, end_exclusive) of the top-level `index:` block."""
    for position, line in enumerate(lines):
        if re.match(r"^index\s*:\s*(#.*)?$", line):
            end = len(lines)
            for offset in range(position + 1, len(lines)):
                candidate = lines[offset]
                if not candidate.strip() or candidate.lstrip().startswith("#"):
                    continue
                if not candidate[:1].isspace():
                    end = offset
                    break
            return position + 1, end
    return None


def pin_collection(
    directory: Path, *, collection_key: str, collection_name: str, collection_path: str | None
) -> None:
    """Record the indexed collection in the folder's config.yaml.

    Edits the affected lines in place instead of re-serialising the document, so
    the comments `init` wrote survive. Missing keys are inserted, and the whole
    file is replaced atomically.
    """
    config_path = directory / CONFIG_FILENAME
    if not config_path.is_file():
        raise ConfigError(f"Cannot pin the collection: no {CONFIG_FILENAME} in {directory}")

    values = {
        "collection_path": collection_path,
        "collection_key": collection_key,
        "collection_name": collection_name,
    }
    lines = config_path.read_text(encoding="utf-8").splitlines()
    bounds = _index_block_bounds(lines)

    if bounds is None:
        lines.append("")
        lines.append("index:")
        lines.extend(f"{_DEFAULT_INDENT}{key}: {_yaml_scalar(values[key])}" for key in _PIN_KEYS)
    else:
        start, end = bounds
        indent = _DEFAULT_INDENT
        for line in lines[start:end]:
            stripped = line.lstrip()
            if stripped and not stripped.startswith("#"):
                indent = line[: len(line) - len(stripped)]
                break

        remaining = set(_PIN_KEYS)
        for position in range(start, end):
            match = re.match(r"^(\s*)([A-Za-z_][\w-]*)\s*:", lines[position])
            if match and match.group(2) in remaining:
                key = match.group(2)
                lines[position] = f"{match.group(1)}{key}: {_yaml_scalar(values[key])}"
                remaining.discard(key)

        # Keys the user deleted from their config are re-added at the top of the block.
        for key in reversed(_PIN_KEYS):
            if key in remaining:
                lines.insert(start, f"{indent}{key}: {_yaml_scalar(values[key])}")

    _write_atomic(config_path, "\n".join(lines) + "\n")
    logger.info("pinned collection %s (%s) in %s", collection_name, collection_key, config_path)


def _write_atomic(path: Path, content: str) -> None:
    """Write via a temporary file in the same directory, then rename over."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ConfigError(f"Cannot write {path}: {exc}") from exc


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
