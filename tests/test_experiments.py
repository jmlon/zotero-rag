"""The experiment log: question-set versions and the cumulative run record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zotero_rag.experiments import (
    ExperimentLogError,
    load_log,
    log_path,
    make_run,
    question_set_info,
    record_runs,
)
from zotero_rag_core.evaluation import (
    EvalAggregate,
    EvalQuestion,
    QuestionSet,
    QuestionSetError,
    RunSettings,
    load_question_set,
)

QUESTIONS = """\
version_id: kg-v1
questions:
  - id: a
    question: How wide is the beam?
    answer: Three.
  - id: b
    question: How deep is the search?
    answer: Three hops.
"""


def write(tmp_path: Path, text: str, name: str = "questions.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- question-set versions -------------------------------------------------


def test_the_version_id_is_read_from_the_top_of_the_file(tmp_path: Path) -> None:
    question_set = load_question_set(write(tmp_path, QUESTIONS))
    assert question_set.version_id == "kg-v1"
    assert [q.id for q in question_set.questions] == ["a", "b"]


def test_a_set_without_a_version_still_loads(tmp_path: Path) -> None:
    text = QUESTIONS.replace("version_id: kg-v1\n", "")
    assert load_question_set(write(tmp_path, text)).version_id is None


def test_a_numeric_version_is_read_as_text(tmp_path: Path) -> None:
    text = QUESTIONS.replace("kg-v1", "2")
    assert load_question_set(write(tmp_path, text)).version_id == "2"


def test_a_version_that_cannot_name_a_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="letters, digits"):
        load_question_set(write(tmp_path, QUESTIONS.replace("kg-v1", "kg/v1")))


def test_a_structured_version_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="version_id"):
        load_question_set(write(tmp_path, QUESTIONS.replace("kg-v1", "[1, 2]")))


def test_the_fingerprint_ignores_comments_and_layout(tmp_path: Path) -> None:
    plain = load_question_set(write(tmp_path, QUESTIONS, "a.yaml"))
    commented = load_question_set(write(tmp_path, "# reviewed\n\n" + QUESTIONS, "b.yaml"))
    assert plain.fingerprint == commented.fingerprint


def test_the_fingerprint_changes_when_a_question_does(tmp_path: Path) -> None:
    before = load_question_set(write(tmp_path, QUESTIONS, "a.yaml"))
    after = load_question_set(write(tmp_path, QUESTIONS.replace("Three hops", "Two"), "b.yaml"))
    assert before.fingerprint != after.fingerprint


# --- the log ---------------------------------------------------------------


def question_set(version: str | None = "kg-v1", answer: str = "Three.") -> QuestionSet:
    return QuestionSet(
        version_id=version,
        questions=[
            EvalQuestion(id="a", question="How wide?", answer=answer),
            EvalQuestion(id="b", question="How deep?", answer="Three hops."),
        ],
    )


def run(qs: QuestionSet | None = None, batch: str = "batch1", chunk_size: int = 1000):  # type: ignore[no-untyped-def]
    qs = qs or question_set()
    return make_run(
        "eval",
        batch,
        question_set_info(qs, Path("questions.yaml"), ["a", "b"]),
        RunSettings(chunk_size=chunk_size, top_k=10, retrieval="hybrid"),
        EvalAggregate(questions=2, mean_score=4.0),
        [],
    )


def test_runs_accumulate_across_calls(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "experiments.json"
    record_runs(path, [run(chunk_size=1000)])
    record_runs(path, [run(chunk_size=500), run(chunk_size=1500)])

    log = load_log(path)
    assert [r.settings.chunk_size for r in log.runs] == [1000, 500, 1500]
    assert len({r.run_id for r in log.runs}) == 3
    # Plain JSON a notebook can load without this package.
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["runs"][0]["question_set"]["version_id"] == "kg-v1"
    assert raw["runs"][0]["settings"]["retrieval"] == "hybrid"


def test_a_subset_run_records_which_questions_ran() -> None:
    info = question_set_info(question_set(), Path("q.yaml"), ["b"])
    assert (info.questions, info.questions_run, info.question_ids) == (2, 1, ["b"])
    whole = question_set_info(question_set(), Path("q.yaml"), ["a", "b"])
    assert whole.question_ids is None


def test_an_edited_set_under_an_old_version_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "experiments.json"
    assert record_runs(path, [run()]) == []

    edited = question_set(answer="Four.")
    warnings = record_runs(path, [run(edited)])

    assert len(warnings) == 1
    assert "kg-v1" in warnings[0] and "new `version_id:`" in warnings[0]
    # Recorded anyway: the fingerprint keeps the two sets apart.
    assert len({r.question_set.fingerprint for r in load_log(path).runs}) == 2


def test_a_new_version_for_the_edited_set_is_not_flagged(tmp_path: Path) -> None:
    path = tmp_path / "experiments.json"
    record_runs(path, [run()])
    assert record_runs(path, [run(question_set(version="kg-v2", answer="Four."))]) == []


def test_an_unreadable_log_is_left_untouched(tmp_path: Path) -> None:
    path = tmp_path / "experiments.json"
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ExperimentLogError, match="left as it is"):
        record_runs(path, [run()])

    assert path.read_text(encoding="utf-8") == "{ not json"
    assert list(tmp_path.iterdir()) == [path]  # no temporary file left behind


def test_each_question_set_has_its_own_log(tmp_path: Path) -> None:
    v1 = question_set_info(question_set("kg-v1"), Path("q.yaml"), ["a", "b"])
    v2 = question_set_info(question_set("kg-v2"), Path("q.yaml"), ["a", "b"])
    assert log_path(tmp_path, v1) == tmp_path / "experiments-questions-kg-v1.json"
    assert log_path(tmp_path, v2) == tmp_path / "experiments-questions-kg-v2.json"


def test_an_unversioned_set_is_logged_by_its_fingerprint(tmp_path: Path) -> None:
    unversioned = question_set(version=None)
    info = question_set_info(unversioned, Path("q.yaml"), ["a", "b"])
    assert log_path(tmp_path, info).name == (
        f"experiments-questions-fp-{unversioned.fingerprint}.json"
    )


def test_a_log_written_before_costs_were_recorded_still_loads(tmp_path: Path) -> None:
    """Runs logged by an older version lack the cost fields; they must not be lost."""
    path = tmp_path / "experiments.json"
    record_runs(path, [run()])
    raw = json.loads(path.read_text(encoding="utf-8"))
    for field in ("query_cost", "query_cost_mean", "query_cost_total", "judge_cost_total"):
        raw["runs"][0]["aggregate"].pop(field)
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert load_log(path).runs[0].aggregate.query_cost is None
