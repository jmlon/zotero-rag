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

from zotero_rag import __version__, cli, index_folder
from zotero_rag.config import Config
from zotero_rag.evaluation import JudgeVerdict, RawVerdict, load_questions
from zotero_rag.fingerprint import StoreFingerprint
from zotero_rag.generation import DocumentText, GeneratedQuestion, RawGeneration
from zotero_rag.indexer import IndexStats
from zotero_rag.planning import IndexPlan, PlannedAttachment, SkippedEntry, SkipReason
from zotero_rag.querying import QueryResult, Source, Usage
from zotero_rag.snapshot import read_snapshot
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
    for command in ("list-collections", "init", "index", "query", "eval", "generate-questions"):
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

    notices: list[index_folder.Notice] = []
    index_folder.check_fingerprint(store, Config(), "...", notices.append)

    recorded = store.read_fingerprint()
    assert recorded is not None
    assert recorded.embedding_model == Config().embedding_model
    assert recorded.embedding_dim == 3  # derived from the data, not assumed
    # Adopting is not silent: the user is told what the record cannot vouch for.
    assert [notice.level for notice in notices] == ["note"]
    assert "--repair" in notices[0].text


def test_check_fingerprint_only_warns_about_changed_chunking(tmp_path: Path) -> None:
    store = _populated_store(tmp_path, Config().embedding_model)
    config = Config()
    config.index.chunk_size = 400
    notices: list[index_folder.Notice] = []
    # Must not raise: every stored vector is still valid, merely shaped differently.
    index_folder.check_fingerprint(store, config, "...", notices.append)
    assert notices and all(notice.level == "warning" for notice in notices)


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
    assert set(payload) == {"answer", "sources", "usage", "retrieval"}
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


# --- `generate-questions` (PRD §11.3.2) ----------------------------------


def _generation_index_folder(tmp_path: Path, documents: int = 1) -> None:
    """An index folder with `documents` indexed items and no question set yet."""
    store = VectorStore(tmp_path / "chroma")
    config = Config()
    for number in range(1, documents + 1):
        store.add(
            ids=[f"ATT{number}:1:0", f"ATT{number}:2:0"],
            embeddings=[[1.0, 0.0, 0.5], [0.9, 0.1, 0.4]],
            documents=[f"Paper {number} introduces a fog node.", f"Paper {number} concludes."],
            metadatas=[
                {
                    "attachment_key": f"ATT{number}",
                    "item_key": f"ITEM{number}",
                    "title": f"Paper Number {number}",
                    "authors": f"Author{number}, Ann",
                    "year": "2016",
                    "page": page,
                    "chunk_index": 0,
                    "embedding_model": config.embedding_model,
                }
                for page in (1, 2)
            ],
        )
    store.write_fingerprint(StoreFingerprint.intended(config, embedding_dim=3))
    (tmp_path / "config.yaml").write_text("index:\n  top_k: 3\n", encoding="utf-8")


def _stub_generator(monkeypatch: pytest.MonkeyPatch, per_call: int = 2) -> list[str]:
    """Replace the generation seam; returns the list of languages it was asked for."""
    languages: list[str] = []

    def generate_for(document: DocumentText, count: int, language: str) -> RawGeneration:
        languages.append(language)
        return RawGeneration(
            questions=[
                GeneratedQuestion(
                    question=f"What does item {document.item_key} propose ({number})?",
                    answer=f"It proposes a fog node: number {number}.",
                    expected_points=["it sits between edge and cloud", "it is close to the data"],
                )
                for number in range(1, per_call + 1)
            ],
            prompt_tokens=1200,
            completion_tokens=150,
        )

    monkeypatch.setattr(cli, "build_generator", lambda config: generate_for)
    return languages


def test_generate_questions_is_listed_in_the_help() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert "generate-questions" in result.output


def test_generate_questions_outside_an_index_folder_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["generate-questions"])
    assert result.exit_code == 1
    assert "config.yaml" in result.output


def test_generate_questions_dry_run_costs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generation_index_folder(tmp_path, documents=2)
    monkeypatch.chdir(tmp_path)

    def must_not_build(config: object, name: str) -> object:
        raise AssertionError("--dry-run reached the generator")

    monkeypatch.setattr(cli, "build_generator", must_not_build)

    result = runner.invoke(cli.app, ["generate-questions", "--dry-run"])

    assert result.exit_code == 0
    assert "Paper Number 1" in result.output and "Paper Number 2" in result.output
    assert "prompt tokens" in result.output
    assert "nothing was written" in result.output
    assert not (tmp_path / "questions.yaml").exists()


def test_generate_questions_writes_a_set_that_eval_can_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end guarantee: generation feeds eval without hand-editing."""
    _generation_index_folder(tmp_path, documents=2)
    monkeypatch.chdir(tmp_path)
    _stub_generator(monkeypatch, per_call=2)

    result = runner.invoke(cli.app, ["generate-questions"])

    assert result.exit_code == 0
    assert "Wrote 4 questions" in result.output
    assert "2 documents" in result.output

    questions = load_questions(tmp_path / "questions.yaml")
    assert len(questions) == 4
    # Provenance is exact: each question names the item its text came from.
    assert {question.expected_items[0] for question in questions} == {"ITEM1", "ITEM2"}
    assert [question.id for question in questions] == [
        "author1-2016-1",
        "author1-2016-2",
        "author2-2016-1",
        "author2-2016-2",
    ]


def test_generate_questions_tells_you_to_edit_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generated set is a draft; the command has to say so, not just the file."""
    _generation_index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_generator(monkeypatch)

    result = runner.invoke(cli.app, ["generate-questions"])

    assert "edit it" in result.output
    assert "zotero-rag eval" in result.output
    assert "EDIT THIS FILE" in (tmp_path / "questions.yaml").read_text(encoding="utf-8")


def test_generate_questions_refuses_to_clobber_an_existing_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generation_index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "questions.yaml").write_text("questions: []\n", encoding="utf-8")

    def must_not_build(config: object, name: str) -> object:
        raise AssertionError("it spent a generation call before refusing")

    monkeypatch.setattr(cli, "build_generator", must_not_build)

    result = runner.invoke(cli.app, ["generate-questions"])

    assert result.exit_code == 1
    assert "already exists" in result.output
    assert "--force" in result.output
    # Refused before spending anything, and left the file untouched.
    assert (tmp_path / "questions.yaml").read_text(encoding="utf-8") == "questions: []\n"


def test_generate_questions_force_replaces_the_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generation_index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "questions.yaml").write_text("questions: []\n", encoding="utf-8")
    _stub_generator(monkeypatch)

    result = runner.invoke(cli.app, ["generate-questions", "--force"])

    assert result.exit_code == 0
    assert len(load_questions(tmp_path / "questions.yaml")) == 2


def test_generate_questions_out_writes_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generation_index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_generator(monkeypatch)

    result = runner.invoke(cli.app, ["generate-questions", "--out", "draft.yaml"])

    assert result.exit_code == 0
    assert (tmp_path / "draft.yaml").exists()
    assert not (tmp_path / "questions.yaml").exists()


def test_generate_questions_language_is_passed_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generation_index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    languages = _stub_generator(monkeypatch)

    runner.invoke(cli.app, ["generate-questions", "--language", "Spanish"])

    assert languages == ["Spanish"]


def test_generate_questions_only_item_narrows_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generation_index_folder(tmp_path, documents=3)
    monkeypatch.chdir(tmp_path)
    _stub_generator(monkeypatch, per_call=1)

    result = runner.invoke(cli.app, ["generate-questions", "--only-item", "ITEM2"])

    assert result.exit_code == 0
    questions = load_questions(tmp_path / "questions.yaml")
    assert [question.expected_items[0] for question in questions] == ["ITEM2"]


def test_generate_questions_only_item_with_an_unindexed_key_explains_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generation_index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)
    _stub_generator(monkeypatch)

    result = runner.invoke(cli.app, ["generate-questions", "--only-item", "NOPE"])

    assert result.exit_code == 1
    assert "NOPE" in result.output
    assert "no indexed attachment" in result.output


def test_generate_questions_limit_caps_the_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generation_index_folder(tmp_path, documents=3)
    monkeypatch.chdir(tmp_path)
    _stub_generator(monkeypatch, per_call=1)

    result = runner.invoke(cli.app, ["generate-questions", "--limit", "2"])

    assert result.exit_code == 0
    assert len(load_questions(tmp_path / "questions.yaml")) == 2


def test_generate_questions_reports_a_document_that_produced_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generation_index_folder(tmp_path, documents=2)
    monkeypatch.chdir(tmp_path)

    def flaky(document: DocumentText, count: int, language: str) -> RawGeneration:
        if document.item_key == "ITEM1":
            raise RuntimeError("the provider said no")
        return RawGeneration(
            questions=[GeneratedQuestion(question="Real?", answer="Yes.", expected_points=["a"])]
        )

    monkeypatch.setattr(cli, "build_generator", lambda config: flaky)

    result = runner.invoke(cli.app, ["generate-questions"])

    assert result.exit_code == 0
    assert "Wrote 1 question" in result.output
    assert "produced nothing" in result.output
    assert "Paper Number 1" in result.output


def test_generate_questions_writes_nothing_when_every_document_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generation_index_folder(tmp_path)
    monkeypatch.chdir(tmp_path)

    def always_fails(document: DocumentText, count: int, language: str) -> RawGeneration:
        raise RuntimeError("no")

    monkeypatch.setattr(cli, "build_generator", lambda config: always_fails)

    result = runner.invoke(cli.app, ["generate-questions"])

    assert result.exit_code == 1
    assert "no usable question" in result.output
    assert not (tmp_path / "questions.yaml").exists()


# --- windowed retrieval through the CLI (M7, §11.4) ----------------------


def test_query_help_documents_the_window_flag() -> None:
    result = runner.invoke(cli.app, ["query", "--help"])
    assert "--window" in result.output
    assert "neighbouring" in result.output


def test_window_defaults_to_the_configured_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    (tmp_path / "config.yaml").write_text(
        "index:\n  top_k: 3\n  retrieval_window: 2\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    payload = json.loads(runner.invoke(cli.app, ["eval", "--json"]).stdout)

    assert payload["settings"]["retrieval_window"] == 2


def test_window_flag_overrides_the_configured_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run's window has to reach the report, or two runs are not comparable."""
    _eval_index_folder(tmp_path, ONE_QUESTION)
    (tmp_path / "config.yaml").write_text(
        "index:\n  top_k: 3\n  retrieval_window: 2\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    payload = json.loads(runner.invoke(cli.app, ["eval", "--window", "0", "--json"]).stdout)

    assert payload["settings"]["retrieval_window"] == 0


def test_eval_settings_line_shows_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    result = runner.invoke(cli.app, ["eval", "--window", "1", "--retrieval", "vector"])

    assert "vector ±1 retrieval" in result.output


def test_a_negative_window_is_rejected() -> None:
    result = runner.invoke(cli.app, ["query", "anything", "--window", "-1"])
    assert result.exit_code != 0


# --- hybrid retrieval through the CLI (M7, §11.4.2) ----------------------


def test_query_help_documents_the_retrieval_flag() -> None:
    result = runner.invoke(cli.app, ["query", "--help"])
    assert "--retrieval" in result.output
    assert "hybrid" in result.output


def test_retrieval_mode_defaults_to_the_configured_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    (tmp_path / "config.yaml").write_text(
        "index:\n  top_k: 3\n  retrieval: hybrid\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    payload = json.loads(runner.invoke(cli.app, ["eval", "--json"]).stdout)

    assert payload["settings"]["retrieval"] == "hybrid"


def test_retrieval_flag_overrides_the_configured_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    (tmp_path / "config.yaml").write_text(
        "index:\n  top_k: 3\n  retrieval: hybrid\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    payload = json.loads(runner.invoke(cli.app, ["eval", "--retrieval", "vector", "--json"]).stdout)

    assert payload["settings"]["retrieval"] == "vector"


def test_an_unknown_retrieval_mode_is_rejected() -> None:
    result = runner.invoke(cli.app, ["query", "anything", "--retrieval", "magic"])
    assert result.exit_code != 0


def test_hybrid_builds_the_keyword_index_once_per_eval_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebuilding it per question would tokenize the whole store N times over."""
    _eval_index_folder(tmp_path, ONE_QUESTION + "  - id: two\n    question: And?\n    answer: x\n")
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)
    builds: list[int] = []

    real = cli.KeywordIndex.from_store

    def counted(store: object) -> object:
        builds.append(1)
        return real(store)  # type: ignore[arg-type]

    monkeypatch.setattr(cli.KeywordIndex, "from_store", staticmethod(counted))

    result = runner.invoke(cli.app, ["eval", "--retrieval", "hybrid", "--json"])

    assert result.exit_code == 0
    assert len(json.loads(result.stdout)["results"]) == 2
    assert builds == [1]


def test_vector_mode_builds_no_keyword_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    def must_not_build(store: object) -> object:
        raise AssertionError("vector retrieval built a keyword index")

    monkeypatch.setattr(cli.KeywordIndex, "from_store", staticmethod(must_not_build))

    assert runner.invoke(cli.app, ["eval", "--retrieval", "vector"]).exit_code == 0


def test_eval_settings_line_names_the_retrieval_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    result = runner.invoke(cli.app, ["eval", "--retrieval", "hybrid"])

    assert "hybrid" in result.output


def test_hybrid_is_the_shipped_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§11.4.2 accepted it on the evidence; the packaged config must agree."""
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    payload = json.loads(runner.invoke(cli.app, ["eval", "--json"]).stdout)

    assert payload["settings"]["retrieval"] == "hybrid"
    assert Config().index.retrieval == "hybrid"


# --- reranking through the CLI (M7, §11.4.3) -----------------------------


def test_query_help_documents_the_reranker_flag() -> None:
    result = runner.invoke(cli.app, ["query", "--help"])
    assert "--reranker" in result.output
    assert "cohere" in result.output


def test_reranking_is_off_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaulting it on would break `query` for everyone without a Cohere key."""
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    def must_not_build(config: object, name: str) -> object:
        raise AssertionError("a reranker was built without being asked for")

    monkeypatch.setattr(index_folder, "build_reranker", must_not_build)

    payload = json.loads(runner.invoke(cli.app, ["eval", "--json"]).stdout)

    assert payload["settings"]["reranker"] is None
    assert Config().reranker is None


def test_asking_for_a_reranker_without_a_key_fails_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    result = runner.invoke(cli.app, ["eval", "--reranker", "cohere"])

    assert result.exit_code == 1
    assert "COHERE_API_KEY" in result.output


def test_an_unknown_reranker_is_named_in_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    result = runner.invoke(cli.app, ["query", "anything", "--reranker", "magic"])

    assert result.exit_code == 1
    assert "magic" in result.output


def test_reranker_none_overrides_a_configured_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--reranker none` must switch it off without needing the credential."""
    _eval_index_folder(tmp_path, ONE_QUESTION)
    (tmp_path / "config.yaml").write_text(
        "index:\n  top_k: 3\n  reranker: cohere\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    result = runner.invoke(cli.app, ["eval", "--reranker", "none", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["settings"]["reranker"] is None


def test_a_configured_reranker_is_recorded_in_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)
    monkeypatch.setattr(
        index_folder, "build_reranker", lambda config, name: lambda q, docs, n: [(0, 0.9)]
    )

    result = runner.invoke(cli.app, ["eval", "--reranker", "cohere", "--json"])

    settings = json.loads(result.stdout)["settings"]
    assert settings["reranker"] == "cohere"
    assert settings["reranker_model"] == "rerank-v4.0-fast"
    assert settings["reranker_fanout"] == 4
    assert json.loads(result.stdout)["results"][0]["reranked"] is True


def test_an_openrouter_reranker_is_recorded_by_its_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)
    built: list[str] = []

    def build(config: object, name: str) -> object:
        built.append(name)
        return lambda q, docs, n: [(0, 0.9)]

    monkeypatch.setattr(index_folder, "build_reranker", build)

    result = runner.invoke(cli.app, ["eval", "--reranker", "voyageai/rerank-3", "--json"])

    assert result.exit_code == 0, result.output
    assert built == ["voyageai/rerank-3"]
    settings = json.loads(result.stdout)["settings"]
    assert settings["reranker"] == settings["reranker_model"] == "voyageai/rerank-3"


def test_a_reranker_that_silently_fell_back_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two reports must not look identical when one was only half reranked."""
    _eval_index_folder(tmp_path, ONE_QUESTION)
    monkeypatch.chdir(tmp_path)
    _stub_seams(monkeypatch)

    def flaky(question: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        raise RuntimeError("503")

    monkeypatch.setattr(index_folder, "build_reranker", lambda config, name: flaky)

    result = runner.invoke(cli.app, ["eval", "--reranker", "cohere"])

    assert result.exit_code == 0
    assert "reranking applied to only 0 of 1 questions" in result.output


# --- folder membership and the collection snapshot (PRD-CursesUI §4) ---------


def _collection(key: str, name: str, parent: str | None = None) -> dict[str, object]:
    return {"key": key, "data": {"key": key, "name": name, "parentCollection": parent or False}}


def _paper_with_note(key: str) -> list[dict[str, object]]:
    """An item plus a child note: indexable with no download at all."""
    return [
        {"key": key, "data": {"key": key, "itemType": "journalArticle", "title": f"Paper {key}"}},
        {
            "key": f"N{key}",
            "data": {
                "key": f"N{key}",
                "itemType": "note",
                "parentItem": key,
                "note": f"<p>Notes on paper {key}.</p>",
                "version": 5,
            },
        },
    ]


def _fake_library(
    monkeypatch: pytest.MonkeyPatch, contents: dict[str, list[dict[str, object]]]
) -> None:
    """A pinned tree NET > {ROUTING, TRANSPORT > QUIC}, holding `contents`."""

    class Embeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 1.0, 0.5] for text in texts]

    collections = [
        _collection("NET", "Networking"),
        _collection("ROUTING", "Routing", "NET"),
        _collection("TRANSPORT", "Transport", "NET"),
        _collection("QUIC", "QUIC", "TRANSPORT"),
        _collection("OTHER", "Unrelated"),
    ]
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(cli, "build_client", lambda config: object())
    monkeypatch.setattr(cli, "fetch_collections", lambda zot: collections)
    monkeypatch.setattr(cli, "fetch_username", lambda zot: "someone")
    monkeypatch.setattr(cli, "fetch_collection_items", lambda zot, key: contents.get(key, []))
    monkeypatch.setattr(cli, "build_embeddings", lambda config: Embeddings())


def test_index_records_folder_membership_and_writes_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text("index:\n  collection_key: NET\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    shared = _paper_with_note("P1")
    _fake_library(
        monkeypatch,
        {"NET": [], "ROUTING": shared, "QUIC": [*shared, *_paper_with_note("P2")]},
    )

    result = runner.invoke(cli.app, ["index"])
    assert result.exit_code == 0, result.output

    store = VectorStore(tmp_path / "chroma")
    assert store.collection_keys_by_attachment() == {
        "NP1": ["QUIC", "ROUTING"],  # one item, two folders
        "NP2": ["QUIC"],
    }
    snapshot = read_snapshot(tmp_path)
    assert snapshot is not None
    assert [root.key for root in snapshot.roots] == ["NET"]  # the pinned tree only
    assert snapshot.all_keys() == {"NET", "ROUTING", "TRANSPORT", "QUIC"}
    # Written with the chunks, so there was nothing to bring up to date.
    assert "folder membership" not in result.output


def test_index_backfills_a_store_written_before_membership_existed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text("index:\n  collection_key: NET\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _fake_library(monkeypatch, {"ROUTING": _paper_with_note("P1")})
    assert runner.invoke(cli.app, ["index"]).exit_code == 0

    # Make the store look like one indexed by 0.4.x: chunks, no keys, no snapshot.
    store = VectorStore(tmp_path / "chroma")
    old = store.all_chunks()
    store.delete_ids([chunk.id for chunk in old])
    store.add(
        ids=[chunk.id for chunk in old],
        embeddings=[[1.0, 1.0, 1.0]] * len(old),
        documents=[chunk.text for chunk in old],
        metadatas=[
            {k: v for k, v in chunk.metadata.items() if k != "collection_keys"} for chunk in old
        ],
    )
    (tmp_path / "collections.json").unlink()

    result = runner.invoke(cli.app, ["index"])

    assert result.exit_code == 0, result.output
    assert "Updated the folder membership of 1 attachment (no re-embedding)" in result.output
    assert VectorStore(tmp_path / "chroma").collection_keys_by_attachment() == {"NP1": ["ROUTING"]}
    assert read_snapshot(tmp_path) is not None


def test_a_dry_run_writes_no_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text("index:\n  collection_key: NET\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _fake_library(monkeypatch, {"ROUTING": _paper_with_note("P1")})
    assert runner.invoke(cli.app, ["index", "--dry-run"]).exit_code == 0
    assert read_snapshot(tmp_path) is None


def test_skips_the_user_can_fix_name_the_items_and_say_how() -> None:
    plan = IndexPlan(
        skipped=[
            SkippedEntry(key="F6CRQDYK", title="Exploring LLMs", reason=SkipReason.NOT_IN_STORAGE),
            SkippedEntry(key="L1", title="A local file", reason=SkipReason.LINKED_FILE),
        ]
    )
    out = StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli, "console", Console(file=out, width=200, no_color=True))
        cli._print_skips(plan)
    text = out.getvalue()
    assert "file not uploaded to Zotero storage" in text
    assert "Exploring LLMs (F6CRQDYK)" in text
    assert "Settings → Sync" in text
    assert "Convert Linked Files to Stored Files" in text
