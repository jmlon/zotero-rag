"""The evaluation harness: question loading, metrics, judging, aggregation.

No API access: the grading seam is a local callable, exactly as the answering
seam is in `test_querying`. The golden sets shipped in `questions/` are parsed
here too, so a typo in one of them fails the suite rather than a live run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zotero_rag.config import Config
from zotero_rag.errors import ZoteroRagError
from zotero_rag.evaluation import (
    EVIDENCE_COVERAGE,
    SCORES,
    Coverage,
    EvalQuestion,
    Grade,
    JudgeVerdict,
    QuestionOutcome,
    QuestionSetError,
    RawVerdict,
    aggregate,
    cited_expected,
    evaluate,
    evaluate_question,
    evidence_coverage,
    evidence_found,
    evidence_hit,
    judge_answer,
    load_questions,
    parse_questions,
    resolve_questions_path,
    retrieval_hit,
    run_settings,
)
from zotero_rag.querying import QueryResult, Source, Usage
from zotero_rag.vectorstore import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_questions(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- loading -------------------------------------------------------------


def test_loads_a_mapping_with_a_questions_list(tmp_path: Path) -> None:
    path = write_questions(
        tmp_path,
        """
questions:
  - id: one
    question: What is a fog node?
    expected_points:
      - it sits between the edge and the cloud
    expected_items: [ABC123]
""",
    )
    questions = load_questions(path)
    assert len(questions) == 1
    assert questions[0].id == "one"
    assert questions[0].expected_points == ["it sits between the edge and the cloud"]
    # A single key written as a scalar is the obvious thing to write.
    assert questions[0].expected_items == ["ABC123"]


def test_loads_a_bare_top_level_list(tmp_path: Path) -> None:
    path = write_questions(tmp_path, "- question: Anything?\n  answer: Something.\n")
    assert load_questions(path)[0].id == "q1"  # ids are positional when absent


def test_missing_file_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="No question set at"):
        load_questions(tmp_path / "absent.yaml")


def test_invalid_yaml_names_the_file(tmp_path: Path) -> None:
    path = write_questions(tmp_path, "questions:\n  - [unclosed\n")
    with pytest.raises(QuestionSetError, match="Invalid YAML"):
        load_questions(path)


def test_an_empty_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="empty"):
        load_questions(write_questions(tmp_path, "\n"))


def test_a_document_without_questions_says_so(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="no `questions:` key"):
        load_questions(write_questions(tmp_path, "settings:\n  top_k: 5\n"))


def test_a_question_with_no_text_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(QuestionSetError, match="no `question:` text"):
        parse_questions([{"id": "x", "answer": "y"}], tmp_path / "q.yaml")


def test_a_question_with_nothing_to_grade_against_is_rejected(tmp_path: Path) -> None:
    """Without points or a reference answer the judge has no reference at all."""
    with pytest.raises(QuestionSetError, match="nothing to grade it against"):
        parse_questions([{"question": "Why?"}], tmp_path / "q.yaml")


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    """Ids key the report, so a duplicate would silently overwrite a row."""
    entries = [
        {"id": "same", "question": "a", "answer": "x"},
        {"id": "same", "question": "b", "answer": "y"},
    ]
    with pytest.raises(QuestionSetError, match="duplicate question id"):
        parse_questions(entries, tmp_path / "q.yaml")


def test_a_malformed_expectation_names_the_question(tmp_path: Path) -> None:
    entries = [{"id": "bad", "question": "a", "answer": "x", "expected_items": {"key": 1}}]
    with pytest.raises(QuestionSetError, match="'bad'"):
        parse_questions(entries, tmp_path / "q.yaml")


def test_ground_truth_carries_both_prose_and_points() -> None:
    question = EvalQuestion(
        id="q", question="?", answer="A prose reference.", expected_points=["first", "second"]
    )
    truth = question.ground_truth
    assert "A prose reference." in truth
    # Numbered, because the judge reports coverage by index.
    assert "1. first" in truth and "2. second" in truth


def test_ground_truth_of_a_points_only_question_has_no_empty_prose_block() -> None:
    question = EvalQuestion(id="q", question="?", expected_points=["only this"])
    assert question.ground_truth.startswith("Expected points:")


# --- where the question set lives ----------------------------------------


def test_a_relative_questions_file_resolves_inside_the_index_folder(tmp_path: Path) -> None:
    config = Config()
    config.source_dir = tmp_path
    assert resolve_questions_path(config, None) == tmp_path / "questions.yaml"


def test_an_explicit_questions_path_wins(tmp_path: Path) -> None:
    config = Config()
    config.source_dir = tmp_path
    explicit = Path("/elsewhere/set.yaml")
    assert resolve_questions_path(config, explicit) == explicit


def test_an_absolute_configured_path_is_left_alone(tmp_path: Path) -> None:
    config = Config()
    config.source_dir = tmp_path
    config.eval.questions_file = "/shared/golden.yaml"
    assert resolve_questions_path(config, None) == Path("/shared/golden.yaml")


# --- retrieval metrics ---------------------------------------------------


def source(
    number: int = 1,
    item_key: str = "ITEM1",
    title: str = "A Paper About Fog",
    cited: bool = False,
) -> Source:
    return Source(number=number, item_key=item_key, title=title, cited=cited)


def test_a_hit_is_any_retrieved_chunk_of_an_expected_item() -> None:
    question = EvalQuestion(id="q", question="?", answer="x", expected_items=["ITEM2"])
    assert retrieval_hit([source(item_key="ITEM1"), source(item_key="ITEM2")], question) is True
    assert retrieval_hit([source(item_key="ITEM1")], question) is False


def test_a_title_expectation_matches_case_insensitively_as_a_substring() -> None:
    question = EvalQuestion(id="q", question="?", answer="x", expected_titles=["about FOG"])
    assert retrieval_hit([source(title="A Paper About Fog")], question) is True
    assert retrieval_hit([source(title="Something else")], question) is False


def test_a_question_naming_no_item_reports_no_hit_either_way() -> None:
    """Reporting `False` would misstate the index for a question that never asked."""
    question = EvalQuestion(id="q", question="?", answer="x")
    assert retrieval_hit([source()], question) is None
    assert cited_expected([source(cited=True)], question) is None


def test_citing_an_expected_item_is_stricter_than_retrieving_it() -> None:
    question = EvalQuestion(id="q", question="?", answer="x", expected_items=["ITEM1"])
    retrieved_only = [source(item_key="ITEM1", cited=False)]
    assert retrieval_hit(retrieved_only, question) is True
    assert cited_expected(retrieved_only, question) is False
    assert cited_expected([source(item_key="ITEM1", cited=True)], question) is True


# --- the judge -----------------------------------------------------------


# --- passage hits ----------------------------------------------------------

QUOTE = "The beam width is set to 3 and the maximum depth to 3 for all datasets."


def chunk_source(*excerpts: str) -> Source:
    return Source(number=1, item_key="ITEM1", excerpts=list(excerpts))


def test_a_quote_inside_a_retrieved_chunk_is_a_passage_hit() -> None:
    question = EvalQuestion(id="q", question="?", answer="x", expected_evidence=[QUOTE])
    chunk = f"We run every experiment the same way. {QUOTE} Results follow."
    assert evidence_hit([chunk_source(chunk)], question) is True


def test_case_punctuation_and_line_breaks_do_not_matter() -> None:
    reflowed = "the beam\nwidth is set to 3, and the maximum depth to 3 for all data-\nsets"
    assert evidence_coverage(QUOTE, [reflowed]) >= EVIDENCE_COVERAGE


def test_a_quote_split_across_two_retrieved_chunks_still_counts() -> None:
    half = len(QUOTE) // 2
    chunks = [f"Earlier text. {QUOTE[:half]}", f"{QUOTE[half:]} Later text."]
    assert evidence_found(QUOTE, chunks)
    # Only one of the two halves reached the model: the answer did not.
    assert not evidence_found(QUOTE, chunks[:1])


def test_another_passage_of_the_right_paper_is_not_a_passage_hit() -> None:
    question = EvalQuestion(
        id="q", question="?", answer="x", expected_items=["ITEM1"], expected_evidence=[QUOTE]
    )
    sources = [chunk_source("The method explores relation paths with an LLM.")]
    assert retrieval_hit(sources, question) is True
    assert evidence_hit(sources, question) is False


def test_every_quote_must_be_retrieved() -> None:
    other = "Pruning with Sentence-BERT cuts the number of LLM calls in half."
    question = EvalQuestion(id="q", question="?", answer="x", expected_evidence=[QUOTE, other])
    assert evidence_hit([chunk_source(QUOTE)], question) is False
    assert evidence_hit([chunk_source(QUOTE, other)], question) is True


def test_a_question_without_evidence_has_no_passage_hit() -> None:
    question = EvalQuestion(id="q", question="?", answer="x")
    assert evidence_hit([chunk_source(QUOTE)], question) is None


def test_expected_evidence_is_read_from_the_file(tmp_path: Path) -> None:
    path = write_questions(
        tmp_path,
        "questions:\n"
        "  - id: q\n"
        "    question: How wide is the beam?\n"
        "    answer: Three.\n"
        f'    expected_evidence: ["{QUOTE}"]\n',
    )
    assert load_questions(path)[0].expected_evidence == [QUOTE]


def test_the_passage_hit_rate_ignores_questions_without_evidence() -> None:
    outcomes = [
        QuestionOutcome(id="a", question="?", grade=Grade.GOOD, score=3, passage_hit=True),
        QuestionOutcome(id="b", question="?", grade=Grade.GOOD, score=3, passage_hit=False),
        QuestionOutcome(id="c", question="?", grade=Grade.GOOD, score=3),
    ]
    assert aggregate(outcomes).passage_hit_rate == 0.5
    assert aggregate(outcomes[2:]).passage_hit_rate is None


def verdict_fn(
    grade: str = "good",
    covered: list[int] | None = None,
    partial: list[int] | None = None,
    tokens: tuple[int, int] = (100, 20),
):  # type: ignore[no-untyped-def]
    """A grading seam that always returns the same verdict."""

    def call(question: str, ground_truth: str, answer: str) -> RawVerdict:
        return RawVerdict(
            verdict=JudgeVerdict(
                grade=grade,  # type: ignore[arg-type]
                covered_points=covered or [],
                partial_points=partial or [],
                rationale="because",
            ),
            prompt_tokens=tokens[0],
            completion_tokens=tokens[1],
        )

    return call


THREE_POINTS = EvalQuestion(
    id="q", question="?", answer="ref", expected_points=["one", "two", "three"]
)


def test_a_verdict_becomes_a_grade_and_a_coverage_count() -> None:
    judgement = judge_answer(THREE_POINTS, "an answer", verdict_fn("good", covered=[1, 2]))
    assert judgement.grade is Grade.GOOD
    assert judgement.score == SCORES[Grade.GOOD]
    assert judgement.coverage == Coverage(total=3, covered=2, partial=0)
    assert judgement.coverage.score == pytest.approx(2 / 3)


def test_a_partial_point_counts_half() -> None:
    judgement = judge_answer(THREE_POINTS, "a", verdict_fn(covered=[1], partial=[2]))
    assert judgement.coverage.score == pytest.approx((1 + 0.5) / 3)


def test_a_hallucinated_point_index_cannot_push_coverage_over_100_percent() -> None:
    """The judge is a language model; an out-of-range index is dropped."""
    judgement = judge_answer(THREE_POINTS, "a", verdict_fn(covered=[1, 2, 3, 4, 99, 0, -1]))
    assert judgement.coverage == Coverage(total=3, covered=3, partial=0)
    assert judgement.coverage.score == 1.0


def test_a_point_named_both_covered_and_partial_counts_once() -> None:
    judgement = judge_answer(THREE_POINTS, "a", verdict_fn(covered=[1], partial=[1, 2]))
    assert judgement.coverage == Coverage(total=3, covered=1, partial=1)


def test_coverage_of_a_question_without_points_is_not_zero_but_absent() -> None:
    """Averaging an absent coverage in as zero would understate every run."""
    assert Coverage(total=0).score is None


def test_the_judge_is_retried_and_then_reported_as_an_error() -> None:
    """A grading outage must not be recorded as a retrieval regression."""
    calls: list[int] = []

    def failing(question: str, ground_truth: str, answer: str) -> RawVerdict:
        calls.append(1)
        raise RuntimeError("503 from the provider")

    judgement = judge_answer(THREE_POINTS, "an answer", failing, attempts=3)

    assert len(calls) == 3
    assert judgement.grade is Grade.ERROR
    assert judgement.score == 0
    assert "503" in judgement.rationale
    # The points are still counted as expected, so coverage reads 0 of 3, not 0 of 0.
    assert judgement.coverage == Coverage(total=3)


def test_a_transient_judge_failure_is_recovered() -> None:
    attempts: list[int] = []
    working = verdict_fn("perfect", covered=[1, 2, 3])

    def flaky(question: str, ground_truth: str, answer: str) -> RawVerdict:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("timeout")
        return working(question, ground_truth, answer)

    judgement = judge_answer(THREE_POINTS, "an answer", flaky)
    assert judgement.grade is Grade.PERFECT
    assert len(attempts) == 2


# --- running one question ------------------------------------------------


def result_with(sources: list[Source], answer: str = "An answer [1].") -> QueryResult:
    return QueryResult(
        question="?",
        answer=answer,
        sources=sources,
        usage=Usage(prompt_tokens=500, completion_tokens=60, embedding_tokens=9),
        chunks_retrieved=len(sources),
    )


def test_an_answered_question_records_retrieval_and_grading_together() -> None:
    question = EvalQuestion(
        id="q", question="?", answer="ref", expected_points=["one", "two"], expected_items=["ITEM1"]
    )
    outcome = evaluate_question(
        question,
        lambda _: result_with([source(item_key="ITEM1", cited=True)]),
        verdict_fn("good", covered=[1]),
    )

    assert outcome.grade is Grade.GOOD
    assert outcome.hit is True and outcome.cited_expected is True
    assert outcome.cited_sources == 1
    assert outcome.coverage == Coverage(total=2, covered=1)
    # Grading tokens are kept apart from answering tokens.
    assert (outcome.prompt_tokens, outcome.completion_tokens) == (500, 60)
    assert outcome.judge_tokens == 120


def test_retrieving_nothing_is_no_answer_and_spends_no_judge_call() -> None:
    calls: list[int] = []

    def must_not_be_called(question: str, ground_truth: str, answer: str) -> RawVerdict:
        calls.append(1)
        raise AssertionError("the judge was called with nothing to grade")

    outcome = evaluate_question(THREE_POINTS, lambda _: result_with([]), must_not_be_called)

    assert outcome.grade is Grade.NO_ANSWER
    assert outcome.score == 0
    assert not calls


def test_a_question_that_cannot_be_answered_is_reported_not_raised() -> None:
    """One timeout must not cost the other twenty rows (PRD §10)."""

    def failing(question: str) -> QueryResult:
        raise ZoteroRagError("the language model call failed")

    outcome = evaluate_question(THREE_POINTS, failing, verdict_fn())

    assert outcome.grade is Grade.ERROR
    assert "language model call failed" in outcome.error


def test_a_failing_context_assessment_does_not_fail_the_row() -> None:
    def broken(question: str, ground_truth: str, context: str) -> str:
        raise RuntimeError("no")

    outcome = evaluate_question(
        THREE_POINTS, lambda _: result_with([source()]), verdict_fn(), broken
    )
    assert outcome.grade is Grade.GOOD
    assert outcome.context_sufficiency is None


def test_the_context_assessment_sees_the_retrieved_excerpts() -> None:
    seen: list[str] = []

    def assess(question: str, ground_truth: str, context: str) -> str:
        seen.append(context)
        return "sufficient"

    passage = Source(number=1, item_key="ITEM1", excerpts=["fog nodes sit at the edge"])
    outcome = evaluate_question(
        THREE_POINTS, lambda _: result_with([passage]), verdict_fn(), assess
    )

    assert outcome.context_sufficiency == "sufficient"
    assert "fog nodes sit at the edge" in seen[0]


def test_evaluate_runs_every_question_in_file_order() -> None:
    questions = [
        EvalQuestion(id="a", question="first?", answer="x"),
        EvalQuestion(id="b", question="second?", answer="y"),
    ]
    started: list[str] = []
    outcomes = evaluate(
        questions,
        lambda _: result_with([source()]),
        verdict_fn(),
        on_start=lambda question: started.append(question.id),
    )
    assert [outcome.id for outcome in outcomes] == ["a", "b"]
    assert started == ["a", "b"]


def test_parallel_evaluation_overlaps_questions_and_keeps_file_order() -> None:
    import threading
    import time

    questions = [EvalQuestion(id=name, question=f"{name}?", answer="x") for name in "abcd"]
    # Every answer waits until all four are in flight, so a sequential run would
    # time out here; the first question also finishes last.
    barrier = threading.Barrier(4, timeout=5)

    def ask(question: str) -> QueryResult:
        barrier.wait()
        if question == "a?":
            time.sleep(0.05)
        return result_with([source()])

    outcomes = evaluate(questions, ask, verdict_fn(), workers=4)

    assert [outcome.id for outcome in outcomes] == ["a", "b", "c", "d"]


def test_an_exception_from_the_progress_hook_stops_a_parallel_run() -> None:
    class Stop(Exception):
        pass

    def on_start(question: EvalQuestion) -> None:
        raise Stop

    questions = [EvalQuestion(id=name, question=f"{name}?", answer="x") for name in "ab"]
    with pytest.raises(Stop):
        evaluate(
            questions, lambda _: result_with([source()]), verdict_fn(), on_start=on_start, workers=2
        )


# --- aggregation ---------------------------------------------------------


def outcome(
    identifier: str,
    grade: Grade,
    hit: bool | None = None,
    cited: bool | None = None,
    coverage: Coverage | None = None,
) -> QuestionOutcome:
    return QuestionOutcome(
        id=identifier,
        question="?",
        grade=grade,
        score=SCORES[grade],
        hit=hit,
        cited_expected=cited,
        coverage=coverage or Coverage(),
    )


def test_aggregate_averages_scores_and_counts_grades() -> None:
    outcomes = [
        outcome("a", Grade.PERFECT),
        outcome("b", Grade.PARTIAL),
        outcome("c", Grade.WRONG),
    ]
    summary = aggregate(outcomes)
    assert summary.questions == 3
    assert summary.mean_score == pytest.approx((5 + 2 + 0) / 3)
    assert summary.grade_counts == {"perfect": 1, "partial": 1, "wrong": 1}


def test_hit_rate_covers_only_the_questions_that_named_an_item() -> None:
    outcomes = [
        outcome("a", Grade.GOOD, hit=True, cited=True),
        outcome("b", Grade.GOOD, hit=False, cited=False),
        outcome("c", Grade.GOOD),  # named no item: excluded, not counted as a miss
    ]
    summary = aggregate(outcomes)
    assert summary.hit_rate == pytest.approx(0.5)
    assert summary.citation_rate == pytest.approx(0.5)


def test_metrics_with_no_applicable_question_stay_absent() -> None:
    summary = aggregate([outcome("a", Grade.GOOD)])
    assert summary.hit_rate is None
    assert summary.citation_rate is None
    assert summary.mean_coverage is None
    assert summary.graded_questions == 0


def test_mean_coverage_ignores_questions_without_points() -> None:
    outcomes = [
        outcome("a", Grade.GOOD, coverage=Coverage(total=2, covered=2)),
        outcome("b", Grade.GOOD, coverage=Coverage(total=4, covered=1)),
        outcome("c", Grade.GOOD, coverage=Coverage()),
    ]
    summary = aggregate(outcomes)
    assert summary.mean_coverage == pytest.approx((1.0 + 0.25) / 2)
    assert summary.graded_questions == 2


def test_a_grading_outage_is_counted_apart_from_a_wrong_answer() -> None:
    summary = aggregate([outcome("a", Grade.ERROR), outcome("b", Grade.WRONG)])
    assert summary.errors == 1
    assert summary.grade_counts == {"wrong": 1, "error": 1}


def test_aggregating_nothing_does_not_divide_by_zero() -> None:
    assert aggregate([]).questions == 0


def test_total_tokens_spans_answering_grading_and_embedding() -> None:
    row = QuestionOutcome(
        id="a",
        question="?",
        grade=Grade.GOOD,
        score=4,
        prompt_tokens=100,
        completion_tokens=10,
        embedding_tokens=5,
        judge_tokens=50,
    )
    assert aggregate([row]).total_tokens == 165


# --- comparability -------------------------------------------------------


def test_run_settings_record_everything_that_moves_the_numbers(tmp_path: Path) -> None:
    """A report without these is a set of figures with no "compared to what"."""
    store = VectorStore(tmp_path / "chroma")
    store.add(
        ids=["A:1:0"],
        embeddings=[[1.0, 0.0, 0.5]],
        documents=["text"],
        metadatas=[{"attachment_key": "A", "page": 1}],
    )
    config = Config()
    config.index.collection_name = "Networking"
    config.index.collection_key = "SHWKWYDD"

    settings = run_settings(config, store, top_k=7, questions_path=tmp_path / "set.yaml")

    assert settings.collection_key == "SHWKWYDD"
    assert settings.top_k == 7
    assert settings.llm_model == config.llm_model
    assert settings.judge_model == config.judge_model
    assert settings.chunk_size == config.index.chunk_size
    assert settings.store_chunks == 1
    # Derived from a stored vector, so it cannot drift from the data.
    assert settings.embedding_dim == 3
    assert settings.retrieval == "vector"


def test_the_judge_model_defaults_to_the_answering_model() -> None:
    config = Config()
    assert config.judge_model == config.llm_model
    config.eval.judge_model = "openai/gpt-5"
    assert config.judge_model == "openai/gpt-5"


# --- the shipped golden sets ---------------------------------------------


@pytest.mark.parametrize(("name", "expected_questions"), [("Writing", 3), ("Networking", 21)])
def test_the_shipped_golden_sets_parse(name: str, expected_questions: int) -> None:
    """A typo in a golden set should fail here, not part-way through a live run."""
    questions = load_questions(REPO_ROOT / "questions" / f"{name}.yaml")
    assert len(questions) == expected_questions
    for question in questions:
        assert question.expected_points, f"{question.id} has no points to score"
        assert question.has_expected_source, f"{question.id} names no expected item"
        assert question.ground_truth.strip()


def test_the_networking_set_maps_three_questions_to_each_of_its_seven_articles() -> None:
    """One source article per question is what makes hit rate discriminating."""
    questions = load_questions(REPO_ROOT / "questions" / "Networking.yaml")
    per_item: dict[str, int] = {}
    for question in questions:
        assert len(question.expected_items) == 1, question.id
        per_item[question.expected_items[0]] = per_item.get(question.expected_items[0], 0) + 1
    assert len(per_item) == 7
    assert set(per_item.values()) == {3}
    # The saved Bing search page that looks like the CFR-RL paper is not a hit.
    assert "KJFG9YUE" not in per_item


def test_a_parallel_run_reports_its_wall_clock_time_not_the_sum() -> None:
    outcomes = [
        QuestionOutcome(id=name, question="?", grade=Grade.GOOD, score=3, elapsed_s=4.0)
        for name in "abcd"
    ]
    assert aggregate(outcomes).elapsed_s == 16.0
    assert aggregate(outcomes, elapsed_s=4.5).elapsed_s == 4.5


def test_question_time_is_reported_as_median_and_quartiles() -> None:
    """One stalled request must not move the typical time."""
    times = [2.0, 2.5, 3.0, 3.5, 146.0]
    outcomes = [
        QuestionOutcome(id=str(n), question="?", grade=Grade.GOOD, score=3, elapsed_s=t)
        for n, t in enumerate(times)
    ]
    spread = aggregate(outcomes).question_time

    assert spread is not None
    assert spread.median_s == 3.0
    assert (spread.p25_s, spread.p75_s) == (2.5, 3.5)


def test_a_single_question_has_a_degenerate_spread() -> None:
    outcome = QuestionOutcome(id="a", question="?", grade=Grade.GOOD, score=3, elapsed_s=4.0)
    spread = aggregate([outcome]).question_time
    assert spread is not None
    assert (spread.p25_s, spread.median_s, spread.p75_s) == (4.0, 4.0, 4.0)
    assert aggregate([]).question_time is None
