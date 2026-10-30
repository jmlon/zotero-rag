"""`compare-embeddings`: one store per model × chunking, scored by the eval harness.

No network: Zotero, the embedding client, the answering model and the judge are
all stubbed, while chunking, the stores and retrieval run for real.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from zotero_rag import cli
from zotero_rag.config import Config, EmbeddingModelSpec, EmbeddingPricing, load_config
from zotero_rag.embedding import DEFAULT_BATCH_SIZE, build_embeddings
from zotero_rag.embedding_comparison import (
    chunkings,
    context_warning,
    model_slug,
    plan_variants,
    scaled_overlap,
    variant_config,
)
from zotero_rag.evaluation import JudgeVerdict, RawVerdict
from zotero_rag.fingerprint import StoreFingerprint
from zotero_rag.indexer import DIMENSION_PROBE
from zotero_rag.vectorstore import VectorStore

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_env", lambda: None)
    for name in ("ZOTERO_LIBRARY_ID", "ZOTERO_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "250")


# --- planning ------------------------------------------------------------


def _config(tmp_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200) -> Config:
    config = Config()
    config.index.chunk_size = chunk_size
    config.index.chunk_overlap = chunk_overlap
    config.source_dir = tmp_path
    return config


def test_model_slug_is_one_safe_folder_name() -> None:
    assert model_slug("openai/text-embedding-3-small") == "openai_text-embedding-3-small"
    assert model_slug("nvidia/nemotron-3.5-embed-1b:free") == "nvidia_nemotron-3.5-embed-1b_free"


def test_overlap_keeps_the_configured_ratio() -> None:
    config = Config()  # 1000 / 200
    assert scaled_overlap(config, 1000) == 200
    assert scaled_overlap(config, 500) == 100
    assert scaled_overlap(config, 2000) == 400


def test_default_overlaps_follow_the_ratio(tmp_path: Path) -> None:
    pairs, impossible = chunkings(_config(tmp_path), [500, 1000, 2000], None)
    assert pairs == [(500, 100), (1000, 200), (2000, 400)]
    assert impossible == []


def test_given_overlaps_pair_with_every_size_and_impossible_ones_are_set_apart(
    tmp_path: Path,
) -> None:
    pairs, impossible = chunkings(_config(tmp_path), [500, 1000], [0, 200, 500])
    assert pairs == [(500, 0), (500, 200), (1000, 0), (1000, 200), (1000, 500)]
    assert impossible == [(500, 500)]


def test_the_baseline_comes_first_and_reads_chroma(tmp_path: Path) -> None:
    config = _config(tmp_path)
    variants = plan_variants(config, ["baai/bge-m3"], [(500, 100)])

    assert [(v.embedding_model, v.chunk_size, v.baseline) for v in variants] == [
        ("openai/text-embedding-3-small", 1000, True),
        ("baai/bge-m3", 500, False),
    ]
    assert variants[0].store_dir == tmp_path / "chroma"
    assert variants[1].store_dir == tmp_path / "embeddings" / "baai_bge-m3" / "500-100"
    assert variants[1].chunk_overlap == 100


def test_the_default_model_at_another_overlap_is_a_variant(tmp_path: Path) -> None:
    config = _config(tmp_path)
    variants = plan_variants(config, [config.embedding_model], [(1000, 200), (1000, 0)])

    assert [(v.chunk_overlap, v.baseline) for v in variants] == [(200, True), (0, False)]
    assert variants[1].store_dir.name == "1000-0"


def test_the_baseline_combination_is_never_planned_twice(tmp_path: Path) -> None:
    config = _config(tmp_path)
    variants = plan_variants(
        config,
        [config.embedding_model, config.embedding_model, "baai/bge-m3"],
        [(1000, 200), (500, 100)],
    )
    pairs = [(v.embedding_model, v.chunk_size) for v in variants]
    assert len(pairs) == len(set(pairs)) == 4
    assert sum(v.baseline for v in variants) == 1
    # The default model at another size is a variant, not the baseline.
    other_size = next(
        v for v in variants if v.embedding_model == config.embedding_model and not v.baseline
    )
    assert other_size.store_dir.parent.name == model_slug(config.embedding_model)


def test_variant_config_changes_only_model_and_chunking(tmp_path: Path) -> None:
    config = _config(tmp_path)
    variant = plan_variants(config, ["baai/bge-m3"], [(500, 100)])[1]
    settings = variant_config(config, variant)

    assert settings.embedding_model == "baai/bge-m3"
    assert (settings.index.chunk_size, settings.index.chunk_overlap) == (500, 100)
    assert settings.llm_model == config.llm_model
    assert config.embedding_model == "openai/text-embedding-3-small"  # not mutated


def test_a_chunk_beyond_the_context_window_is_flagged(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.embedding_models = [EmbeddingModelSpec(id="tiny/model", context_window=512)]
    small, large = plan_variants(config, ["tiny/model"], [(1000, 200), (4000, 800)])[1:]
    assert context_warning(config, small) is None
    assert "512-token window" in (context_warning(config, large) or "")


def test_the_catalogue_is_loaded_from_the_packaged_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    spec = config.embedding_model_spec("openai/text-embedding-3-small")
    assert spec is not None and spec.cost(1_000_000) == pytest.approx(0.02)
    assert config.embedding_model_spec("unknown/model") is None


# --- the command -----------------------------------------------------------


QUESTIONS = """
questions:
  - id: fog
    question: What is a fog node?
    expected_points: [it sits between the edge devices and the cloud]
    expected_items: [ITEM1]
"""

CATALOGUE = """
index:
  collection_key: NET
  top_k: 3
embedding_models:
  - id: cheap/model
    pricing: {input: 0.01}
  - id: broken/model
    pricing: {input: 0.5}
"""


def _index_folder(tmp_path: Path) -> None:
    """A pinned, indexed folder: one baseline chunk and a one-question golden set."""
    (tmp_path / "config.yaml").write_text(CATALOGUE, encoding="utf-8")
    (tmp_path / "questions.yaml").write_text(QUESTIONS, encoding="utf-8")
    config = Config()
    store = VectorStore(tmp_path / "chroma")
    store.add(
        ids=["ATT1:1:0"],
        embeddings=[[1.0, 0.0, 0.5]],
        documents=["A fog node sits between the edge devices and the cloud."],
        metadatas=[
            {
                "attachment_key": "ATT1",
                "item_key": "ITEM1",
                "title": "What is a Fog Node",
                "page": 1,
                "embedding_model": config.embedding_model,
            }
        ],
    )
    store.write_fingerprint(StoreFingerprint.intended(config, embedding_dim=3))


def _stub_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub every remote seam; returns the models an embedding client was built for."""
    built_for: list[str] = []

    class Embeddings:
        def __init__(self, model: str) -> None:
            self.model = model

        def _vector(self) -> list[float]:
            if self.model == "broken/model":
                raise RuntimeError("model not found")
            return [1.0, 0.0, 0.5]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._vector() for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._vector()

    def build_embeddings(config: Config) -> Embeddings:
        built_for.append(config.embedding_model)
        return Embeddings(config.embedding_model)

    class Chat:
        def invoke(self, messages: object) -> AIMessage:
            return AIMessage(content="It sits between the edge and the cloud [1].")

    def verdict_of(question: str, ground_truth: str, answer: str) -> RawVerdict:
        return RawVerdict(
            verdict=JudgeVerdict(grade="good", covered_points=[1], rationale="Right.")
        )

    paper = [
        {"key": "P1", "data": {"key": "P1", "itemType": "journalArticle", "title": "Fog"}},
        {
            "key": "NP1",
            "data": {
                "key": "NP1",
                "itemType": "note",
                "parentItem": "P1",
                "note": "<p>A fog node sits between the edge and the cloud.</p>",
                "version": 3,
            },
        },
    ]
    collection = {"key": "NET", "data": {"key": "NET", "name": "Net", "parentCollection": False}}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(cli, "build_client", lambda config: object())
    monkeypatch.setattr(cli, "fetch_collections", lambda zot: [collection])
    monkeypatch.setattr(cli, "fetch_username", lambda zot: "someone")
    monkeypatch.setattr(cli, "fetch_collection_items", lambda zot, key: paper)
    monkeypatch.setattr(cli, "build_embeddings", build_embeddings)
    monkeypatch.setattr(cli, "build_chat", lambda config: Chat())
    monkeypatch.setattr(cli, "build_verdict_fn", lambda config: verdict_of)
    return built_for


def test_compare_is_listed_in_the_help() -> None:
    assert "compare-embeddings" in runner.invoke(cli.app, ["--help"]).output


def test_dry_run_plans_and_builds_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_network(monkeypatch)

    result = runner.invoke(cli.app, ["compare-embeddings", "--chunk-size", "500", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "embeddings/cheap_model/500-100" in result.output
    assert "to build" in result.output
    assert "Dry run" in result.output
    assert not (tmp_path / "embeddings").exists()


def test_compare_builds_each_variant_and_scores_all_of_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_network(monkeypatch)

    result = runner.invoke(
        cli.app, ["compare-embeddings", "--model", "cheap/model", "--yes", "--json"]
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    baseline, cheap = report["variants"]
    assert baseline["baseline"] and baseline["store"] == str(tmp_path / "chroma")
    assert cheap["store"] == str(tmp_path / "embeddings" / "cheap_model" / "1000-200")
    assert cheap["built_chunks"] == cheap["store_chunks"] == 1
    assert cheap["index_tokens"] > 0
    assert cheap["index_cost"] == pytest.approx(cheap["index_tokens"] * 0.01 / 1_000_000)
    for variant in (baseline, cheap):
        assert variant["error"] == ""
        assert variant["aggregate"]["mean_score"] == 4
    # The baseline model is not in this folder's catalogue, so it has no price.
    assert baseline["index_cost"] is None

    # The variant's store records what built it, so a later run can trust it.
    fingerprint = VectorStore(Path(cheap["store"])).read_fingerprint()
    assert fingerprint is not None
    assert (fingerprint.embedding_model, fingerprint.chunk_size) == ("cheap/model", 1000)


def test_a_failing_model_is_reported_without_losing_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_network(monkeypatch)

    result = runner.invoke(cli.app, ["compare-embeddings", "--yes"])

    assert result.exit_code == 0, result.output
    assert "could not build broken/model @ 1000/200" in result.output
    assert "broken/model @ 1000/200: could not build: model not found" in result.output
    assert "cheap/model" in result.output and "(baseline)" in result.output


def test_no_build_evaluates_only_existing_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    built_for = _stub_network(monkeypatch)

    result = runner.invoke(
        cli.app,
        ["compare-embeddings", "--model", "cheap/model", "--no-build", "--yes", "--json"],
    )

    assert result.exit_code == 0, result.output
    baseline, cheap = json.loads(result.stdout)["variants"]
    assert baseline["aggregate"] is not None
    assert cheap["aggregate"] is None and "no chunks" in cheap["error"]
    assert "cheap/model" not in built_for
    assert not (tmp_path / "embeddings").exists()


def test_a_second_run_reuses_the_built_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_network(monkeypatch)
    arguments = ["compare-embeddings", "--model", "cheap/model", "--yes", "--json"]

    assert runner.invoke(cli.app, arguments).exit_code == 0
    second = runner.invoke(cli.app, arguments)

    assert second.exit_code == 0, second.output
    cheap = json.loads(second.stdout)["variants"][1]
    assert cheap["built_chunks"] == 0  # nothing new to embed
    assert cheap["store_chunks"] == 1


def test_pricing_rejects_an_unknown_unit() -> None:
    with pytest.raises(ValueError):
        EmbeddingPricing.model_validate({"unit": "per_1K_tokens", "input": 0.1})


def test_overlaps_are_compared_in_their_own_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_network(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "compare-embeddings",
            "--model",
            "cheap/model",
            "--chunk-size",
            "300",
            "--chunk-overlap",
            "0",
            "--chunk-overlap",
            "300",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "skipping chunk size 300 with overlap 300" in result.stderr
    variants = json.loads(result.stdout)["variants"]
    assert [(v["chunk_size"], v["chunk_overlap"]) for v in variants] == [(1000, 200), (300, 0)]
    assert (tmp_path / "embeddings" / "cheap_model" / "300-0").is_dir()
    fingerprint = VectorStore(Path(variants[1]["store"])).read_fingerprint()
    assert fingerprint is not None and fingerprint.chunk_overlap == 0


def _two_models(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        CATALOGUE.replace("broken/model", "other/model"), encoding="utf-8"
    )


def test_each_model_is_built_by_its_own_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index_folder(tmp_path)
    _two_models(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_network(monkeypatch)
    # Both models must be mid-build at once to get past this: built one after the
    # other, the first would wait alone and the barrier would break.
    together = threading.Barrier(2, timeout=5)
    threads: dict[str, str] = {}

    class Embeddings:
        def __init__(self, model: str) -> None:
            self.model = model

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if texts == [DIMENSION_PROBE] and threading.current_thread().name.startswith("embed"):
                threads[self.model] = threading.current_thread().name
                together.wait()
            return [[1.0, 0.0, 0.5] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.5]

    monkeypatch.setattr(cli, "build_embeddings", lambda config: Embeddings(config.embedding_model))

    result = runner.invoke(cli.app, ["compare-embeddings", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    assert set(threads) == {"cheap/model", "other/model"}
    assert len(set(threads.values())) == 2
    variants = json.loads(result.stdout)["variants"]
    assert [v["error"] for v in variants] == ["", "", ""]
    assert all(v["built_chunks"] == 1 for v in variants[1:])


def test_one_worker_builds_the_models_in_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index_folder(tmp_path)
    _two_models(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_network(monkeypatch)
    active = 0
    most_active = 0
    lock = threading.Lock()

    class Embeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            nonlocal active, most_active
            with lock:
                active += 1
                most_active = max(most_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return [[1.0, 0.0, 0.5] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.5]

    monkeypatch.setattr(cli, "build_embeddings", lambda config: Embeddings())

    result = runner.invoke(cli.app, ["compare-embeddings", "--workers", "1", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    assert most_active == 1
    assert all(v["store_chunks"] == 1 for v in json.loads(result.stdout)["variants"])


# --- request batching ---------------------------------------------------------


def test_embedding_requests_are_batched_below_provider_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Google's endpoint rejects more than 250 texts per request; LangChain's own
    # default is 1000, which sent a long paper's 291 chunks in one call.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = Config()
    config.global_.resolve_env()
    config.index.embedding_model = "google/gemini-embedding-001"

    assert build_embeddings(config).chunk_size == DEFAULT_BATCH_SIZE < 250

    config.embedding_models = [EmbeddingModelSpec(id="google/gemini-embedding-001", batch_size=16)]
    assert build_embeddings(config).chunk_size == 16
