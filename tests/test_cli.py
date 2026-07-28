"""CLI surface: help text, version, and clean failure without credentials.

These never reach the network: `load_env` is neutralised so a developer's real
`.env` cannot leak in, and the credential check fails before any API call.
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console
from typer.testing import CliRunner

from zotero_rag import __version__, cli
from zotero_rag.config import Config
from zotero_rag.evaluation import JudgeVerdict, RawVerdict
from zotero_rag.fingerprint import StoreFingerprint
from zotero_rag.indexer import IndexStats
from zotero_rag.planning import IndexPlan, PlannedAttachment
from zotero_rag.querying import QueryResult, Source, Usage
from zotero_rag.vectorstore import VectorStore
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
    for command in ("list-collections", "init", "index", "query", "eval"):
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


@pytest.mark.parametrize("arguments", [["index", "-v", "--dry-run"], ["query", "-v", "a question"]])
def test_verbose_is_accepted_after_the_subcommand(
    arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD §3.3/§3.4 document `-v` on the command itself, not only before it."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, arguments)
    # It gets as far as the missing index folder, so the flag itself parsed.
    assert "No such option" not in result.output
    assert "zotero-rag init" in result.output


def test_index_summary_reports_items_seen(tmp_path: Path) -> None:
    """PRD §10 lists items seen among the required summary fields."""
    plan = IndexPlan(
        collections=["Writing"],
        items_seen=6,
        attachments=[
            PlannedAttachment(
                attachment_key=f"A{n}",
                item_key=f"I{n}",
                title="t",
                authors="",
                year="",
                collection_name="Writing",
            )
            for n in range(4)
        ],
    )
    stats = IndexStats(indexed=4, chunks=146, pages=22)
    out = StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli, "console", Console(file=out, width=200, no_color=True))
        cli._print_index_summary(plan, stats, VectorStore(tmp_path / "chroma"), verbose=False)
    text = out.getvalue()
    assert "Indexed 4 documents" in text
    assert "146 chunks from 22 pages" in text  # all PDFs, so "pages" is right
    assert "6 items seen in 1 collection" in text


@pytest.fixture
def pypdf_logger() -> logging.Logger:
    """The pypdf logger, reset so the assertion cannot depend on test order."""
    logger = logging.getLogger("pypdf")
    logger.setLevel(logging.WARNING)
    return logger


def test_pypdf_recovery_chatter_is_hidden_by_default(
    pypdf_logger: logging.Logger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pypdf warns and recovers on malformed files; that noise broke the progress bar."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(cli.app, ["index", "--dry-run"])
    assert not pypdf_logger.isEnabledFor(logging.WARNING)


@pytest.mark.parametrize("arguments", [["-v", "index", "--dry-run"], ["index", "-v", "--dry-run"]])
def test_verbose_brings_back_pypdf_warnings(
    arguments: list[str],
    pypdf_logger: logging.Logger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Suppression must be one flag away, wherever `-v` is given."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(cli.app, arguments)
    assert pypdf_logger.isEnabledFor(logging.WARNING)


def test_query_requires_a_question() -> None:
    result = runner.invoke(cli.app, ["query"])
    assert result.exit_code != 0


def test_query_outside_an_index_folder_points_at_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion 3, for `query`: fail before any network call."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["query", "¿Qué TIPs hay para escritores?"])
    assert result.exit_code == 1
    assert "zotero-rag init" in result.output


def test_query_in_a_never_indexed_folder_points_at_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner.invoke(cli.app, ["init", "."])
    result = runner.invoke(cli.app, ["query", "anything"])
    assert result.exit_code == 1
    assert "no vector store yet" in result.output
    # It stops before needing any credential.
    assert "OPENROUTER" not in result.output


def _populated_store(directory: Path, model: str, fingerprint: bool = True) -> VectorStore:
    """A store holding one chunk, optionally fingerprinted with `model`."""
    store = VectorStore(directory / "chroma")
    store.add(
        ids=["A:1:0"],
        embeddings=[[1.0, 0.0, 0.5]],
        documents=["some text"],
        metadatas=[{"attachment_key": "A", "page": 1, "embedding_model": model}],
    )
    if fingerprint:
        config = Config()
        config.index.embedding_model = model
        store.write_fingerprint(StoreFingerprint.intended(config, embedding_dim=3))
    return store


def test_query_refuses_when_the_configured_model_is_not_the_stored_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The M4 guard, end to end: the whole point of §11.1."""
    _populated_store(tmp_path, "old/model")
    (tmp_path / "config.yaml").write_text(
        "index:\n  embedding_model: new/model\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli.app, ["query", "anything"])

    assert result.exit_code == 1
    assert "embedding_model" in result.output
    assert "old/model" in result.output and "new/model" in result.output
    assert "--repair" in result.output
    # It refuses before needing any credential, so no key is demanded first.
    assert "OPENROUTER_API_KEY" not in result.output


def test_check_fingerprint_adopts_a_store_that_predates_it(tmp_path: Path) -> None:
    """Re-embedding a good store just to gain a guard would be a poor trade."""
    store = _populated_store(tmp_path, "some/model", fingerprint=False)
    assert store.read_fingerprint() is None

    cli._check_fingerprint(store, Config(), remedy="...")

    recorded = store.read_fingerprint()
    assert recorded is not None
    assert recorded.embedding_model == Config().embedding_model
    assert recorded.embedding_dim == 3  # derived from the data, not assumed


def test_check_fingerprint_only_warns_about_changed_chunking(tmp_path: Path) -> None:
    store = _populated_store(tmp_path, Config().embedding_model)
    config = Config()
    config.index.chunk_size = 400
    # Must not raise: every stored vector is still valid, merely shaped differently.
    cli._check_fingerprint(store, config, remedy="...")


def test_refetch_requires_force() -> None:
    """Clearing the cache without rebuilding would only cost downloads."""
    result = runner.invoke(cli.app, ["index", "--refetch"])
    assert result.exit_code == 1
    assert "only applies with --force" in result.output


def test_index_help_documents_the_new_reindex_flags() -> None:
    result = runner.invoke(cli.app, ["index", "--help"])
    for flag in ("--force", "--refetch", "--repair", "--prune"):
        assert flag in result.output


def _sample_result(answer: str = "Escribir todos los días [1].", cited: bool = True) -> QueryResult:
    return QueryResult(
        question="¿Qué TIPs hay para escritores?",
        answer=answer,
        sources=[
            Source(
                number=1,
                item_key="ITEM1",
                attachment_key="ATT1",
                title="Writing Tips",
                authors="Doe, Jane",
                year="2021",
                zotero_url="https://www.zotero.org/u/items/ITEM1",
                page=4,
                cited=cited,
                excerpts=["Write every day, even badly."],
            )
        ],
        usage=Usage(prompt_tokens=120, completion_tokens=30, embedding_tokens=9),
        chunks_retrieved=1,
    )


def _capture_answer(
    show_sources: bool, verbose: bool = False, result: QueryResult | None = None
) -> str:
    out = StringIO()
    console = Console(file=out, width=200, no_color=True, highlight=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli, "console", console)
        cli._print_answer(result or _sample_result(), show_sources, verbose)
    return out.getvalue()


def test_answer_is_followed_by_a_numbered_source_list() -> None:
    text = _capture_answer(show_sources=False)
    assert "Escribir todos los días [1]." in text
    assert "[1] Writing Tips — Doe, Jane (2021), p. 4" in text
    assert "https://www.zotero.org/u/items/ITEM1" in text
    # The chunk text is only shown on request.
    assert "Write every day" not in text


def test_show_sources_prints_the_retrieved_chunk_text() -> None:
    assert "Write every day, even badly." in _capture_answer(show_sources=True)


def test_an_uncited_answer_lists_no_sources() -> None:
    """An "I don't know" must not sit above a list implying those sources backed it."""
    text = _capture_answer(show_sources=False, result=_sample_result(answer="No sé.", cited=False))
    assert "No sé." in text
    assert "Writing Tips" not in text
    assert "cites no sources" in text


def test_show_sources_still_reveals_uncited_retrievals() -> None:
    text = _capture_answer(show_sources=True, result=_sample_result(answer="No sé.", cited=False))
    assert "Writing Tips" in text
    assert "Write every day, even badly." in text


def test_verbose_reports_prompt_and_completion_tokens() -> None:
    text = _capture_answer(show_sources=False, verbose=True)
    assert "120 prompt + 30 completion tokens" in text


def test_json_output_has_the_documented_shape() -> None:
    payload = json.loads(cli._answer_as_json(_sample_result(), show_sources=False))
    assert set(payload) == {"answer", "sources", "usage"}
    assert payload["answer"].startswith("Escribir")
    assert payload["usage"]["total_tokens"] == 150
    source = payload["sources"][0]
    assert source["number"] == 1
    assert source["page"] == 4
    assert source["zotero_url"].endswith("ITEM1")
    # Chunk text is omitted unless asked for, keeping the default payload small.
    assert "excerpts" not in source


def test_json_output_includes_excerpts_with_show_sources() -> None:
    payload = json.loads(cli._answer_as_json(_sample_result(), show_sources=True))
    assert payload["sources"][0]["excerpts"] == ["Write every day, even badly."]


def test_index_requires_openrouter_before_doing_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real run needs an embedding key; say so before downloading anything."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(cli.app, ["init", "."])
    result = runner.invoke(cli.app, ["index"])
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in result.output


def test_dry_run_does_not_require_openrouter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run makes no embedding calls, so it must not demand that key."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(cli.app, ["init", "."])
    result = runner.invoke(cli.app, ["index", "--dry-run"])
    assert result.exit_code == 1
    # It gets past the OpenRouter check and stops at the Zotero credentials.
    assert "ZOTERO" in result.output
    assert "OPENROUTER" not in result.output


def test_index_rejects_both_collection_selectors() -> None:
    result = runner.invoke(
        cli.app, ["index", "--dry-run", "--collection", "A", "--collection-key", "K"]
    )
    assert result.exit_code == 1
    assert "not both" in result.output


def test_index_outside_an_index_folder_points_at_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Must fail on the missing config.yaml, before any network call."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["index", "--dry-run"])
    assert result.exit_code == 1
    assert "zotero-rag init" in result.output


def test_init_works_offline_without_any_credentials(tmp_path: Path) -> None:
    """Acceptance criterion 2: init contacts nothing, so no keys are needed."""
    target = tmp_path / "idx"
    result = runner.invoke(cli.app, ["init", str(target)])
    assert result.exit_code == 0, result.output
    assert (target / "config.yaml").is_file()
    assert (target / ".env").is_file()
    assert "created" in result.output


def test_init_tells_you_to_cd_into_the_new_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`index` and `query` only work inside the folder, so the cd must be spelled out."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["init", "Networking"])
    assert result.exit_code == 0, result.output
    assert "1. cd Networking" in result.output
    assert "zotero-rag index --collection" in result.output


def test_init_offers_both_ways_to_name_a_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name is the easier route; the key is the escape hatch when it is ambiguous."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["init", "Networking"])
    assert '--collection "NAME"' in result.output
    assert "--collection-key KEY" in result.output
    assert "case-insensitively" in result.output
    assert "not unique" in result.output


def test_init_in_place_does_not_tell_you_to_cd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["init", "."])
    assert "cd" not in result.output
    assert "1. add your credentials" in result.output


def test_init_mentions_a_parent_env_that_will_be_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the blank template looks like the only place keys can live."""
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["init", "Networking"])
    assert "will be used for any variable you leave blank" in result.output


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


# --- `eval` (PRD §11.3) --------------------------------------------------


def _eval_index_folder(tmp_path: Path, questions: str) -> None:
    """An index folder holding one indexed chunk and a golden set."""
    store = VectorStore(tmp_path / "chroma")
    config = Config()
    store.add(
        ids=["ATT1:1:0"],
        embeddings=[[1.0, 0.0, 0.5]],
        documents=["A fog node sits between the edge devices and the cloud."],
        metadatas=[
            {
                "attachment_key": "ATT1",
                "item_key": "ITEM1",
                "title": "What is a Fog Node",
                "authors": "Tordera, Eva",
                "year": "2016",
                "page": 3,
                "embedding_model": config.embedding_model,
            }
        ],
    )
    store.write_fingerprint(StoreFingerprint.intended(config, embedding_dim=3))
    (tmp_path / "config.yaml").write_text("index:\n  top_k: 3\n", encoding="utf-8")
    (tmp_path / "questions.yaml").write_text(questions, encoding="utf-8")


ONE_QUESTION = """
questions:
  - id: fog
    question: What is a fog node?
    expected_points:
      - it sits between the edge devices and the cloud
      - it provides compute closer to the data source
    expected_items: [ITEM1]
"""


def _stub_seams(monkeypatch: pytest.MonkeyPatch, grade: str = "good") -> None:
    """Replace the three network seams `eval` builds, leaving retrieval real."""

    class Embeddings:
        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.5]

    class Chat:
        def invoke(self, messages: object) -> AIMessage:
            return AIMessage(content="A fog node sits between the edge and the cloud [1].")

    def verdict_of(question: str, ground_truth: str, answer: str) -> RawVerdict:
        return RawVerdict(
            verdict=JudgeVerdict(
                grade=grade,  # type: ignore[arg-type]
                covered_points=[1],
                partial_points=[2],
                rationale="States the position but not the compute benefit.",
            ),
            prompt_tokens=310,
            completion_tokens=25,
        )

    monkeypatch.setattr(cli, "build_embeddings", lambda config: Embeddings())
    monkeypatch.setattr(cli, "build_chat", lambda config: Chat())
    monkeypatch.setattr(cli, "build_verdict_fn", lambda config: verdict_of)


def test_eval_is_listed_in_the_help() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert "eval" in result.output


def test_eval_outside_an_index_folder_points_at_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["eval"])
    assert result.exit_code == 1
    assert "config.yaml" in result.output


def test_eval_without_a_question_set_says_where_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    (tmp_path / "questions.yaml").unlink()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 1
    assert "No question set at" in result.output
    assert "questions.yaml" in result.output


def test_eval_reports_the_table_and_one_summary_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    result = runner.invoke(cli.app, ["eval"])

    assert result.exit_code == 0
    assert "fog" in result.output
    assert "good" in result.output
    # One point covered, one partial, of two: 1.5/2 = 75%.
    assert "1+1/2" in result.output
    assert "mean score" in result.output
    assert "retrieval hit 100%" in result.output
    assert "point coverage 75%" in result.output
    # The settings that produced the numbers, so two runs can be compared.
    assert "top_k 3" in result.output
    assert Config().embedding_model in result.output


def test_eval_json_is_machine_readable_and_uncontaminated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    result = runner.invoke(cli.app, ["eval", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["settings"]["top_k"] == 3
    assert payload["aggregate"]["hit_rate"] == 1.0
    assert payload["aggregate"]["questions"] == 1
    row = payload["results"][0]
    assert row["id"] == "fog"
    assert row["grade"] == "good"
    assert row["cited_expected"] is True
    # The fraction travels with the counts rather than being re-derived.
    assert row["coverage"]["score"] == 0.75


def test_eval_top_k_override_is_recorded_in_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberately poor top_k must be visible in what the run reports."""
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    result = runner.invoke(cli.app, ["eval", "--top-k", "1", "--json"])

    assert json.loads(result.stdout)["settings"]["top_k"] == 1


def test_eval_only_runs_the_named_question(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _eval_index_folder(
        tmp_path, ONE_QUESTION + "  - id: other\n    question: Why?\n    answer: x\n"
    )
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    result = runner.invoke(cli.app, ["eval", "--only", "other", "--json"])

    payload = json.loads(result.stdout)
    assert [row["id"] for row in payload["results"]] == ["other"]


def test_eval_only_with_an_unknown_id_lists_what_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    result = runner.invoke(cli.app, ["eval", "--only", "nope"])

    assert result.exit_code == 1
    assert "nope" in result.output
    assert "fog" in result.output


def test_eval_assess_context_adds_the_retrieval_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)
    monkeypatch.setattr(
        cli, "build_context_assessor", lambda config: lambda q, truth, context: "partial"
    )

    result = runner.invoke(cli.app, ["eval", "--assess-context", "--json"])

    payload = json.loads(result.stdout)
    assert payload["results"][0]["context_sufficiency"] == "partial"
    assert payload["aggregate"]["context_counts"] == {"partial": 1}
