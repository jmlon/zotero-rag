"""Comparing embedding models and chunk sizes against one question set.

An embedding model and a chunk size are both fixed when a store is built, so the
only honest way to compare them is one store per combination, each scored by the
same `eval` harness with the same answering model and judge. The configured model
at the configured chunk size and overlap is the baseline and lives where it always
has, in `chroma/`; every other combination gets its own store under
`embeddings/<model>/<chunk_size>-<chunk_overlap>/`, so the baseline is never touched and a variant
that has been built once is reused by every later comparison.

Cost is reported as embedding tokens over the whole store, priced from the
`embedding_models` catalogue, because that is what switching model or chunk size
would actually cost to index. The answering tokens per question are reported
beside it: a larger chunk at the same `top_k` makes every query more expensive,
which an embedding price alone would hide.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from zotero_rag.config import Config
from zotero_rag.embedding import count_tokens
from zotero_rag.evaluation import EvalAggregate, QuestionOutcome
from zotero_rag.vectorstore import CHROMA_DIRNAME, VectorStore

EMBEDDINGS_DIRNAME = "embeddings"

#: Characters per token, roughly, for English prose under cl100k. Used only to warn
#: that a chunk size may not fit a model's context window.
_CHARS_PER_TOKEN = 4


def model_slug(model_id: str) -> str:
    """A folder name for a model id: `openai/text-embedding-3-small` → `openai_text-...`.

    Slashes would nest a folder per provider and `:` is not allowed on every
    filesystem, so anything outside a safe set becomes `_`.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def scaled_overlap(config: Config, chunk_size: int) -> int:
    """The overlap for `chunk_size`, at the same ratio as the configured pair.

    A fixed 200-character overlap would be 40% of a 500-character chunk and 10% of
    a 2000-character one, which would change two things at once. At the configured
    size the configured overlap is returned exactly.
    """
    if chunk_size == config.index.chunk_size:
        return config.index.chunk_overlap
    ratio = config.index.chunk_overlap / config.index.chunk_size
    return min(round(chunk_size * ratio), chunk_size - 1)


class Variant(BaseModel):
    """One embedding model at one chunking, and the store that holds it."""

    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    store_dir: Path
    #: The configured model and chunking, read from `chroma/` and never rebuilt.
    baseline: bool = False

    @property
    def label(self) -> str:
        return f"{self.embedding_model} @ {self.chunk_size}/{self.chunk_overlap}"


def variant_store_dir(
    index_dir: Path, embedding_model: str, chunk_size: int, chunk_overlap: int
) -> Path:
    """`embeddings/<model>/<chunk_size>-<chunk_overlap>/` inside the index folder."""
    return (
        index_dir
        / EMBEDDINGS_DIRNAME
        / model_slug(embedding_model)
        / f"{chunk_size}-{chunk_overlap}"
    )


def chunkings(
    config: Config, chunk_sizes: list[int], chunk_overlaps: list[int] | None
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The (size, overlap) pairs to compare, and the impossible ones left out.

    With no overlaps given, each size gets `scaled_overlap`; with overlaps, every
    size is paired with every overlap. A pair whose overlap is not smaller than its
    size cannot be chunked, so it is returned apart for the caller to report.
    """
    pairs: list[tuple[int, int]] = []
    invalid: list[tuple[int, int]] = []
    for size in dict.fromkeys(chunk_sizes):
        overlaps = [scaled_overlap(config, size)] if chunk_overlaps is None else chunk_overlaps
        for overlap in dict.fromkeys(overlaps):
            (pairs if overlap < size else invalid).append((size, overlap))
    return pairs, invalid


def plan_variants(config: Config, models: list[str], pairs: list[tuple[int, int]]) -> list[Variant]:
    """Every model × (chunk size, overlap), baseline first and always present.

    The baseline is included even when not asked for: a table of variants with
    nothing to compare them against answers no question.
    """
    if config.source_dir is None:
        raise ValueError("Variants live in an index folder; this config has none")
    index_dir = config.source_dir
    baseline = Variant(
        embedding_model=config.embedding_model,
        chunk_size=config.index.chunk_size,
        chunk_overlap=config.index.chunk_overlap,
        store_dir=index_dir / CHROMA_DIRNAME,
        baseline=True,
    )
    variants = [baseline]
    seen = {(baseline.embedding_model, baseline.chunk_size, baseline.chunk_overlap)}
    for model in dict.fromkeys(models):
        for size, overlap in dict.fromkeys(pairs):
            if (model, size, overlap) in seen:
                continue
            seen.add((model, size, overlap))
            variants.append(
                Variant(
                    embedding_model=model,
                    chunk_size=size,
                    chunk_overlap=overlap,
                    store_dir=variant_store_dir(index_dir, model, size, overlap),
                )
            )
    return variants


def variant_config(config: Config, variant: Variant) -> Config:
    """The config a run against `variant` would have had.

    Everything but the model and the chunking is shared, which is what makes the
    rows comparable. The fingerprint and `build_embeddings` read from this.
    """
    copied = config.model_copy(deep=True)
    copied.index.embedding_model = variant.embedding_model
    copied.index.chunk_size = variant.chunk_size
    copied.index.chunk_overlap = variant.chunk_overlap
    return copied


def stored_tokens(store: VectorStore) -> int:
    """Embedding tokens over every chunk in the store: what indexing it cost.

    Counted from the stored text rather than remembered from the build, so a store
    built in an earlier run — the baseline included — is costed the same way.
    """
    return count_tokens([chunk.text for chunk in store.all_chunks()])


def context_warning(config: Config, variant: Variant) -> str | None:
    """A note when a chunk is likely longer than the model accepts."""
    spec = config.embedding_model_spec(variant.embedding_model)
    if spec is None or spec.context_window is None:
        return None
    estimated = variant.chunk_size // _CHARS_PER_TOKEN
    if estimated <= spec.context_window:
        return None
    return (
        f"{variant.label}: a {variant.chunk_size}-character chunk is ~{estimated} tokens, "
        f"over {variant.embedding_model}'s {spec.context_window}-token window; "
        "the provider may truncate or reject it."
    )


# --- the report ----------------------------------------------------------


class VariantResult(BaseModel):
    """One row of the comparison: how the store was built, what it scored, its cost."""

    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    store: str
    baseline: bool = False
    provider: str = ""
    embedding_dim: int | None = None
    store_chunks: int = 0
    #: Tokens over every chunk in the store, i.e. the cost of indexing it.
    index_tokens: int = 0
    #: `None` when the model is not in the `embedding_models` catalogue.
    price_per_million: float | None = None
    #: Chunks this run embedded into the store (0 when it was already built).
    built_chunks: int = 0
    build_failures: int = 0
    aggregate: EvalAggregate | None = None
    results: list[QuestionOutcome] = Field(default_factory=list)
    #: Why this variant has no score, when it has none.
    error: str = ""

    @property
    def label(self) -> str:
        return f"{self.embedding_model} @ {self.chunk_size}/{self.chunk_overlap}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def index_cost(self) -> float | None:
        if self.price_per_million is None:
            return None
        return self.index_tokens * self.price_per_million / 1_000_000

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_embedding_cost(self) -> float | None:
        """Embedding the questions themselves: small, but part of the bill."""
        if self.price_per_million is None or self.aggregate is None:
            return None
        return self.aggregate.embedding_tokens * self.price_per_million / 1_000_000

    @computed_field  # type: ignore[prop-decorator]
    @property
    def answer_tokens_per_question(self) -> float | None:
        """Prompt plus completion tokens of the answering model, per question.

        Shown because chunk size moves it: the same `top_k` of larger chunks is a
        longer prompt on every query, for as long as the index is used.
        """
        if self.aggregate is None or not self.aggregate.questions:
            return None
        spent = self.aggregate.prompt_tokens + self.aggregate.completion_tokens
        return spent / self.aggregate.questions


class ComparisonSettings(BaseModel):
    """What every row shares, so the rows differ only in model and chunking."""

    collection_name: str | None = None
    collection_key: str | None = None
    llm_model: str = ""
    judge_model: str = ""
    top_k: int = 0
    retrieval: str = "vector"
    retrieval_window: int = 0
    reranker: str | None = None
    questions_file: str = ""
    questions: int = 0


class ComparisonReport(BaseModel):
    settings: ComparisonSettings
    variants: list[VariantResult] = Field(default_factory=list)
