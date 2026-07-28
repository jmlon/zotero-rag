"""Creating a new index folder.

`init` is deliberately offline: it contacts neither Zotero nor OpenRouter, pins
no collection, and creates no vector store. It only lays down the files the user
needs to fill in before the first `index` run.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from zotero_rag.config import (
    CONFIG_FILENAME,
    ENV_FILENAME,
    GlobalConfig,
    packaged_defaults_text,
)
from zotero_rag.errors import ZoteroRagError

logger = logging.getLogger(__name__)

GITIGNORE_FILENAME = ".gitignore"

GITIGNORE_CONTENT = """\
# Written by `zotero-rag init`. The .env holds API keys; the rest is rebuildable.
.env
chroma/
cache/
"""


class InitResult(BaseModel):
    """What `init` wrote, and what it left alone."""

    directory: Path
    created: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


def render_env_template(global_config: GlobalConfig) -> str:
    """Build a `.env` template using the variable names the config expects.

    Generated from the config rather than hard-coded, so a customised
    `*_env` name in config.yaml produces a matching template.
    """
    return f"""\
# Credentials for zotero-rag. Never commit this file.

# Zotero Web API — create a key at https://www.zotero.org/settings/keys
# The library ID is the numeric userID shown on that page, not your username.
{global_config.zotero_library_id_env}=
{global_config.zotero_api_key_env}=

# OpenRouter — create a key at https://openrouter.ai/keys
# Used by `index` and `query`; `list-collections` does not need it.
{global_config.endpoint_env}={global_config.endpoint}
{global_config.api_key_env}=

# Cohere — OPTIONAL, leave blank unless you set `reranker: cohere` in config.yaml.
# Reranking measurably improves answers and sends less text to the LLM, but it is a
# second provider: https://dashboard.cohere.com/api-keys
{global_config.cohere_api_key_env}=
"""


def init_index_folder(directory: Path, global_config: GlobalConfig) -> InitResult:
    """Create `directory` and lay down config.yaml, a .env template, and .gitignore.

    Refuses to run if the folder is already an index (config.yaml present).
    Existing `.env` and `.gitignore` files are never overwritten — the first may
    hold real keys, and the second may belong to a surrounding repository.
    """
    config_path = directory / CONFIG_FILENAME
    if config_path.is_file():
        raise ZoteroRagError(
            f"{directory} is already an index folder ({CONFIG_FILENAME} exists). "
            "Delete it first, or choose another directory."
        )

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ZoteroRagError(f"Cannot create {directory}: {exc}") from exc

    result = InitResult(directory=directory)

    _write(config_path, packaged_defaults_text(), result)

    env_path = directory / ENV_FILENAME
    if env_path.exists():
        # Never clobber a file that may already hold real credentials.
        result.skipped.append(ENV_FILENAME)
        logger.info("%s already exists; leaving it untouched", env_path)
    else:
        _write(env_path, render_env_template(global_config), result)

    gitignore_path = directory / GITIGNORE_FILENAME
    if gitignore_path.exists():
        result.skipped.append(GITIGNORE_FILENAME)
        logger.info("%s already exists; leaving it untouched", gitignore_path)
    else:
        _write(gitignore_path, GITIGNORE_CONTENT, result)

    return result


def _write(path: Path, content: str, result: InitResult) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ZoteroRagError(f"Cannot write {path}: {exc}") from exc
    result.created.append(path.name)
    logger.info("wrote %s", path)
