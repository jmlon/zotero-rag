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
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, model_validator

from zotero_rag.errors import ConfigError, MissingCredentialError

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.yaml"
ENV_FILENAME = ".env"


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
    #: Only needed when a reranker is enabled (§11.4.3). Cohere is a second provider
    #: because OpenRouter proxies chat and embeddings but not reranking.
    cohere_api_key_env: str = "COHERE_API_KEY"
    #: Optional, for `convert --ocr hosted`: Firecrawl Parse also works keyless,
    #: at lower limits, so a missing key is never an error.
    firecrawl_api_key_env: str = "FIRECRAWL_API_KEY"
    zotero_library_type: Literal["user", "group"] = "user"
    defaults: Defaults = Field(default_factory=Defaults)

    # Resolved from the environment by `resolve_env()`; never read from YAML.
    endpoint: str = Field(default="https://openrouter.ai/api/v1", exclude=True)
    api_key: str = Field(default="", exclude=True)
    zotero_library_id: str = Field(default="", exclude=True)
    zotero_api_key: str = Field(default="", exclude=True)
    cohere_api_key: str = Field(default="", exclude=True)
    firecrawl_api_key: str = Field(default="", exclude=True)

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
        self.cohere_api_key = os.environ.get(self.cohere_api_key_env, "")
        self.firecrawl_api_key = os.environ.get(self.firecrawl_api_key_env, "")

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

    def require_cohere(self) -> None:
        """Asserted only when a reranker is enabled, never for an ordinary query."""
        if not self.cohere_api_key:
            raise MissingCredentialError([self.cohere_api_key_env], "Cohere reranking")


class IndexConfig(BaseModel):
    """Per-index settings, including the collection pinned by the first `index`."""

    collection_path: str | None = None
    collection_key: str | None = None
    collection_name: str | None = None
    recursive: bool = True

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    #: 10 since 0.4.4. Measured: every metric improved from 5 to 10 in both
    #: retrieval modes on both question sets, and the returns stop by 20 (§11.4.1).
    top_k: int = Field(default=10, gt=0)
    #: Neighbouring chunks fetched either side of each hit (M7, §11.4). 0 disables
    #: windowing, which is the pre-M7 behaviour.
    retrieval_window: int = Field(default=0, ge=0)
    #: `hybrid` fuses BM25 keyword hits with the vector hits (M7, §11.4.2). It is a
    #: query-time choice only — no re-indexing, and the store is untouched. The
    #: default because it measured better on two independent question sets at no
    #: token cost; `vector` remains available and is faster to start on a huge store.
    retrieval: Literal["vector", "hybrid"] = "hybrid"
    #: null => inherit `global.defaults.reranker`. `cohere` reranks the candidates
    #: before the answer is built (§11.4.3); it composes with any `retrieval` mode.
    reranker: Literal["cohere"] | None = None
    reranker_model: str = "rerank-v4.0-fast"
    #: Candidates retrieved per final passage, so the reranker has room to discard.
    #: A fanout of 1 lets it only reorder, which changes nothing about what is sent.
    reranker_fanout: int = Field(default=4, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)

    # Which attachment to index when an item offers several readable ones. Roughly
    # a hundred items in a real library hold both a PDF and an HTML snapshot of the
    # same article, so indexing everything would embed that content twice (§11.2).
    # `markdown` first indexes `convert`'s transcripts instead of their sources.
    attachment_preference: list[str] = Field(
        default_factory=lambda: ["markdown", "pdf", "docx", "odt", "epub", "html", "txt"]
    )
    index_all_attachments: bool = False
    # Notes complement an item's file rather than competing with it.
    index_notes: bool = True

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


class EvalConfig(BaseModel):
    """Settings for `zotero-rag eval` (PRD §11.3)."""

    #: Golden set, resolved relative to the index folder unless absolute.
    questions_file: str = "questions.yaml"
    #: null => grade with the same model that answers. Naming a stronger model
    #: here is the usual reason to set it.
    judge_model: str | None = None
    #: Keep at 0. Necessary but not sufficient for comparable runs: repeat runs
    #: reproduce the grade labels and every mechanical metric, while point coverage
    #: can still move a few points, so a small coverage delta is noise.
    judge_temperature: float = Field(default=0.0, ge=0.0)
    #: Context handed to the optional retrieval assessment is truncated to this
    #: many characters, so a large `top_k` cannot overflow the judge's window.
    context_char_budget: int = Field(default=12000, gt=0)

    # --- `generate-questions` (§11.3.2) ---
    #: null => generate with the answering model.
    generator_model: str | None = None
    questions_per_document: int = Field(default=3, gt=0)
    #: Characters of each document shown to the generator. A long paper is
    #: sampled head-and-tail rather than merely truncated, so the conclusions
    #: reach the model along with the abstract.
    document_char_budget: int = Field(default=30000, gt=0)


class QualityThresholds(BaseModel):
    """When a Markdown transcript counts as faithful to its source (PRD-markdown D4).

    Calibrated in Phase 0 against 24 hand-graded conversions: these values pass
    all 22 good ones and fail both bad ones. A FAIL only keeps a file that
    `shrink-storage` would otherwise trash, so tightening them is always safe.
    """

    #: Share of the source's distinct words (≥ 4 letters) found in the transcript.
    min_recall: float = Field(default=0.85, ge=0.0, le=1.0)
    #: The same, for the weakest page or section: catches a dropped page.
    min_worst_segment_recall: float = Field(default=0.80, ge=0.0, le=1.0)
    #: Share of "words" of 20+ letters: spaces lost ("AlgebraicMethodsfor…").
    max_glued_words: float = Field(default=0.02, ge=0.0, le=1.0)
    #: Replacement characters, `(cid:NN)` codes and control characters.
    max_garbage_ratio: float = Field(default=0.005, ge=0.0, le=1.0)
    #: Letters among the visible characters: low means symbol soup.
    min_letter_ratio: float = Field(default=0.60, ge=0.0, le=1.0)
    #: Stray single letters: "T h e  r e s u l t s".
    max_broken_spacing: float = Field(default=0.05, ge=0.0, le=1.0)
    #: Below this many source words there is nothing to compare against.
    min_baseline_words: int = Field(default=200, ge=1)
    #: Pages or sections shorter than this are left out of the worst-segment check.
    min_segment_words: int = Field(default=50, ge=1)


class MarkdownConfig(BaseModel):
    """Settings for Markdown transcripts (`convert`, `shrink-storage`)."""

    quality: QualityThresholds = Field(default_factory=QualityThresholds)


class EmbeddingPricing(BaseModel):
    """What embedding costs, as the provider quotes it."""

    currency: str = "USD"
    #: The only unit quoted by the providers in the catalogue; anything else would
    #: make every cost in the comparison silently wrong by some factor.
    unit: Literal["per_1M_tokens"] = "per_1M_tokens"
    input: float = Field(ge=0.0)


class EmbeddingModelSpec(BaseModel):
    """One entry of the `embedding_models` catalogue that `compare-embeddings` tries."""

    id: str
    provider: str = ""
    dimensions: int | None = None
    #: Tokens. A chunk longer than this is truncated or rejected by the provider.
    context_window: int | None = None
    pricing: EmbeddingPricing | None = None
    #: Texts per embedding request, for a provider that accepts fewer than the
    #: default (`embedding.DEFAULT_BATCH_SIZE`). `None` uses the default.
    batch_size: int | None = Field(default=None, gt=0)
    notes: str = ""

    def cost(self, tokens: int) -> float | None:
        """The price of embedding `tokens`, or `None` when no price is listed."""
        if self.pricing is None:
            return None
        return tokens * self.pricing.input / 1_000_000


class Config(BaseModel):
    """The merged configuration for a run."""

    model_config = ConfigDict(populate_by_name=True)

    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    index: IndexConfig = Field(default_factory=IndexConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    markdown: MarkdownConfig = Field(default_factory=MarkdownConfig)
    #: Candidate embedding models for `compare-embeddings`, with their prices.
    embedding_models: list[EmbeddingModelSpec] = Field(default_factory=list)

    # Directory the index config was loaded from, when there was one.
    source_dir: Path | None = Field(default=None, exclude=True)

    @property
    def llm_model(self) -> str:
        return self.index.llm_model or self.global_.defaults.llm_model

    @property
    def judge_model(self) -> str:
        """The grading model: the configured one, else whatever answers."""
        return self.eval.judge_model or self.llm_model

    @property
    def generator_model(self) -> str:
        """The question-writing model: the configured one, else whatever answers."""
        return self.eval.generator_model or self.llm_model

    @property
    def reranker(self) -> str | None:
        """The reranker to use, or `None` for no reranking."""
        return self.index.reranker or self.global_.defaults.reranker

    @property
    def embedding_model(self) -> str:
        return self.index.embedding_model or self.global_.defaults.embedding_model

    @property
    def vector_db(self) -> str:
        return self.index.vector_db or self.global_.defaults.vector_db

    def embedding_model_spec(self, model_id: str) -> EmbeddingModelSpec | None:
        """The catalogue entry for `model_id`, if it is listed."""
        return next((spec for spec in self.embedding_models if spec.id == model_id), None)


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
