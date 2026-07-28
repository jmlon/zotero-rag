"""The evaluation harness (PRD §11.3): run a golden set and score it.

Retrieval tuning is a series of hypotheses — a reranker, hybrid search, a larger
`top_k` — and without numbers they get accepted because one answer looked good.
This module turns the questions of §12.1 into something executable.

Three metrics, deliberately measuring different things:

- **Retrieval hit rate** is mechanical: did any chunk of an expected item reach
  `top_k`? No model is asked, so it cannot drift. It isolates retrieval failure
  from generation failure, which is the distinction M7 needs.
- **Point coverage** is judged by an LLM against the expected points, because
  "uno es tan bueno como su último paper" and "previous success does not
  guarantee future success" are the same point and no string match sees that.
- **The grade** is the 5-level ordinal scale of `ragbench.models` (§13), reused
  rather than reinvented so the two harnesses stay comparable.

Both network seams — answering a question, grading an answer — are injected
callables, as in `indexer` and `querying`, so the metrics are tested without any
API access.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, computed_field

from zotero_rag.config import Config
from zotero_rag.errors import ZoteroRagError
from zotero_rag.llm import usage_from_message
from zotero_rag.querying import QueryResult, Source
from zotero_rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

QUESTIONS_FILENAME = "questions.yaml"


class QuestionSetError(ZoteroRagError):
    """The golden set is missing, malformed, or has nothing to grade against."""


class Grade(StrEnum):
    """How well an answer matched the ground truth.

    The first six labels and their scores are `ragbench.models.GradeLabel`.
    `error` is ours: ragbench defaults a failed judge call to `wrong`, which
    reports a grading outage as a retrieval regression. Counting it separately
    keeps a flaky provider from looking like a worse index.
    """

    PERFECT = "perfect"
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"
    WRONG = "wrong"
    NO_ANSWER = "no answer"
    ERROR = "error"


#: Numeric score per grade (5-level ordinal; an absent or ungradeable answer is 0).
SCORES: dict[Grade, int] = {
    Grade.PERFECT: 5,
    Grade.GOOD: 4,
    Grade.PARTIAL: 2,
    Grade.POOR: 1,
    Grade.WRONG: 0,
    Grade.NO_ANSWER: 0,
    Grade.ERROR: 0,
}

MAX_SCORE = 5

#: Retrieval-sufficiency labels, best to worst (`ragbench.models.CONTEXT_LABELS`).
CONTEXT_LABELS = ("sufficient", "partial", "absent")


# --- the golden set ------------------------------------------------------


class EvalQuestion(BaseModel):
    """One entry of `questions.yaml`.

    The ground truth may be given as `expected_points`, as a prose `answer`, or
    as both. Points are what coverage is measured against; prose is what a
    reference answer generated alongside the question naturally looks like, and
    discarding it to fit a schema would throw away the more precise statement.
    """

    id: str
    question: str
    expected_points: list[str] = Field(default_factory=list)
    #: Prose reference answer.
    answer: str = ""
    #: Zotero *item* keys a correct answer should rest on.
    expected_items: list[str] = Field(default_factory=list)
    #: Item titles, matched case-insensitively as substrings. For a hand-written
    #: set where looking up keys is more work than the metric is worth.
    expected_titles: list[str] = Field(default_factory=list)

    @property
    def ground_truth(self) -> str:
        """The reference handed to the judge: prose, points, or both."""
        parts: list[str] = []
        if self.answer.strip():
            parts.append(self.answer.strip())
        if self.expected_points:
            listed = "\n".join(
                f"{number}. {point}" for number, point in enumerate(self.expected_points, start=1)
            )
            parts.append(f"Expected points:\n{listed}")
        return "\n\n".join(parts)

    @property
    def has_expected_source(self) -> bool:
        """Whether this question declares which item should be retrieved."""
        return bool(self.expected_items or self.expected_titles)


def _as_str_list(value: Any, path: Path, question_id: str, field: str) -> list[str]:
    """Coerce a YAML scalar or sequence to a list of strings.

    A single key written as `expected_items: ABCD1234` rather than a list is the
    obvious thing to write and harmless to accept.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(entry) for entry in value]
    raise QuestionSetError(
        f"{path}: question {question_id!r} has a {field} that is neither a string nor a list."
    )


def _question_entries(raw: Any, path: Path) -> list[Any]:
    """The list of question mappings, from either accepted top-level shape."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        entries = raw.get("questions")
        if entries is None:
            raise QuestionSetError(
                f"{path}: no `questions:` key. The file holds either a top-level list "
                "of questions or a mapping with a `questions:` list."
            )
        if not isinstance(entries, list):
            raise QuestionSetError(f"{path}: `questions:` must be a list.")
        return entries
    raise QuestionSetError(f"{path}: expected a list of questions or a mapping with `questions:`.")


def parse_questions(raw: Any, path: Path) -> list[EvalQuestion]:
    """Validate a loaded questions document into `EvalQuestion`s.

    Errors name the file and the offending question, because a golden set is
    edited by hand and a silently dropped question would quietly change every
    number in the report.
    """
    questions: list[EvalQuestion] = []
    seen: set[str] = set()

    for position, entry in enumerate(_question_entries(raw, path), start=1):
        if not isinstance(entry, dict):
            raise QuestionSetError(f"{path}: question {position} is not a mapping.")

        # An id is optional; positional ones are stable as long as nothing is
        # inserted, which is enough for a file that is read top to bottom.
        question_id = str(entry.get("id") or f"q{position}")
        text = str(entry.get("question") or "").strip()
        if not text:
            raise QuestionSetError(f"{path}: question {question_id!r} has no `question:` text.")
        if question_id in seen:
            raise QuestionSetError(
                f"{path}: duplicate question id {question_id!r}. Ids key the report, "
                "so they must be unique."
            )
        seen.add(question_id)

        points = _as_str_list(entry.get("expected_points"), path, question_id, "expected_points")
        answer = str(entry.get("answer") or "").strip()
        if not points and not answer:
            raise QuestionSetError(
                f"{path}: question {question_id!r} has neither `expected_points:` nor an "
                "`answer:`. There would be nothing to grade it against."
            )

        questions.append(
            EvalQuestion(
                id=question_id,
                question=text,
                expected_points=points,
                answer=answer,
                expected_items=_as_str_list(
                    entry.get("expected_items"), path, question_id, "expected_items"
                ),
                expected_titles=_as_str_list(
                    entry.get("expected_titles"), path, question_id, "expected_titles"
                ),
            )
        )

    if not questions:
        raise QuestionSetError(f"{path}: the question set is empty.")
    return questions


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read and validate the golden set at `path`."""
    if not path.is_file():
        raise QuestionSetError(
            f"No question set at {path}. Create it with a `questions:` list of "
            "`question:` / `expected_points:` entries, or point --questions at one."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QuestionSetError(f"Cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise QuestionSetError(f"Invalid YAML in {path}: {exc}") from exc
    if raw is None:
        raise QuestionSetError(f"{path}: the question set is empty.")
    return parse_questions(raw, path)


def resolve_questions_path(config: Config, override: Path | None) -> Path:
    """Where the golden set lives: an explicit path, else the configured one.

    A relative configured path resolves against the index folder, so a folder
    carries its own question set the way it carries its own store.
    """
    if override is not None:
        return override
    configured = Path(config.eval.questions_file)
    if configured.is_absolute() or config.source_dir is None:
        return configured
    return config.source_dir / configured


# --- retrieval metrics ---------------------------------------------------


def source_matches(source: Source, question: EvalQuestion) -> bool:
    """Whether this retrieved source is one the question expected.

    Item keys are exact; titles are case-insensitive substrings so a set can name
    `"Jellyfish"` instead of a key.
    """
    if source.item_key and source.item_key in question.expected_items:
        return True
    title = source.title.casefold()
    return any(expected.casefold() in title for expected in question.expected_titles if expected)


def retrieval_hit(sources: Sequence[Source], question: EvalQuestion) -> bool | None:
    """Did any retrieved passage come from an expected item?

    `None` when the question declares no expected source: reporting a hit rate of
    zero for a question that never named an item would misstate the index.
    """
    if not question.has_expected_source:
        return None
    return any(source_matches(source, question) for source in sources)


def cited_expected(sources: Sequence[Source], question: EvalQuestion) -> bool | None:
    """Did the *answer* cite an expected item?

    Stricter than a hit, and the thing §12.7 actually asks for: a passage that
    reached `top_k` but went uncited did not support the answer.
    """
    if not question.has_expected_source:
        return None
    return any(source.cited and source_matches(source, question) for source in sources)


# --- the judge -----------------------------------------------------------


class JudgeVerdict(BaseModel):
    """Structured output schema for the judge.

    Points come back as 1-based indices rather than restated text: it costs a few
    tokens instead of hundreds, and it cannot fail to match a point through
    paraphrase.
    """

    grade: Literal["perfect", "good", "partial", "poor", "wrong", "no answer"] = Field(
        description="perfect: correctly answers everything asked; good: correct with "
        "a minor omission and nothing fabricated; partial: some asked parts right, "
        "others missing or wrong; poor: mostly wrong with a small correct fragment; "
        "wrong: nothing correct, or contradicts the ground truth on the core of the "
        "question; no answer: the response declines to answer or says it does not know."
    )
    covered_points: list[int] = Field(
        default_factory=list,
        description="1-based indices of the expected points the answer states "
        "correctly and in full. Empty if no points were listed.",
    )
    partial_points: list[int] = Field(
        default_factory=list,
        description="1-based indices of expected points the answer only alludes to "
        "or states incompletely. Never repeat an index from covered_points.",
    )
    rationale: str = Field(
        description="One or two sentences naming what was right, missing, wrong."
    )


JUDGE_SYSTEM_PROMPT = """You are a careful, fair grader for a question-answering \
benchmark over a library of academic documents. Decide how well the SYSTEM ANSWER \
answers the QUESTION, using the GROUND TRUTH as the reference for what is correct.

Method:
1. Break the QUESTION into the parts it actually asks for.
2. For each part, check whether the SYSTEM ANSWER is correct and consistent with \
the GROUND TRUTH. Judge semantic meaning, not wording: paraphrases, reordering, \
and a different language from the ground truth are all fine.
3. The GROUND TRUTH may elaborate beyond what the question asks. Omitting that \
extra detail is NOT an error. Only penalise content missing for a part the \
question explicitly asks for, or content that is incorrect.
4. If the GROUND TRUTH lists numbered expected points, report which of those the \
answer covers fully (covered_points) and which it only touches on \
(partial_points), by their numbers.

Distinguish OMISSION (leaving out non-essential detail — not penalised) from \
CONTRADICTION or FABRICATION (stating something false — penalised). Any fabricated \
or incorrect content caps the grade at 'partial' or below, never 'good' or \
'perfect'. Ignore inline citation markers such as [1] or [2]; they are not part of \
the claim being graded. An answer that declines to answer, or says it does not \
know, is 'no answer' rather than 'wrong' — abstaining is not the same failure as \
being confidently incorrect."""


class RawVerdict(BaseModel):
    """A judge response and what the grading call itself cost."""

    verdict: JudgeVerdict
    prompt_tokens: int = 0
    completion_tokens: int = 0


#: The grading network seam: (question, ground truth, answer) -> verdict.
VerdictFn = Callable[[str, str, str], RawVerdict]

#: The retrieval-sufficiency seam: (question, ground truth, context) -> label.
ContextAssessor = Callable[[str, str, str], str]


class ContextVerdict(BaseModel):
    """Structured output schema for the retrieval-sufficiency assessment."""

    sufficiency: Literal["sufficient", "partial", "absent"] = Field(
        description="sufficient: the context contains everything needed to produce "
        "the ground truth; partial: it supports some asked parts but not all; "
        "absent: it does not support the ground truth at all."
    )
    rationale: str = Field(description="Brief justification.")


CONTEXT_SYSTEM_PROMPT = """You assess RETRIEVAL quality, not answer quality. Given \
a QUESTION, its GROUND TRUTH, and the CONTEXT a retrieval system returned, decide \
whether that context contains the information needed to produce the ground truth.

Judge only what is present in the CONTEXT. Do not use your own knowledge, and do \
not reward context that is merely on-topic — the specific facts the ground truth \
asserts must actually be there.

- sufficient: every part the question asks for is supported by the context; a \
careful reader could write the ground truth from it alone.
- partial: some asked parts are supported and others are missing.
- absent: none of the asked parts are supported, or the context is empty."""


def _structured(config: Config, schema: type[BaseModel]) -> Any:
    """A judge client returning `schema`, alongside the raw message for accounting.

    `include_raw` is what keeps the grading tokens visible: the parsed object
    alone carries no usage metadata, and grading a large set is not free.
    """
    config.global_.require_openrouter()
    chat = ChatOpenAI(
        model=config.judge_model,
        temperature=config.eval.judge_temperature,
        openai_api_base=config.global_.endpoint,
        openai_api_key=SecretStr(config.global_.api_key),
    )
    return chat.with_structured_output(schema, include_raw=True)


def _usage(response: dict[str, Any]) -> tuple[int, int]:
    raw = response.get("raw")
    return usage_from_message(raw) if isinstance(raw, AIMessage) else (0, 0)


def build_verdict_fn(config: Config) -> VerdictFn:
    """The live judge, calling the configured grading model."""
    structured = _structured(config, JudgeVerdict)

    def verdict_of(question: str, ground_truth: str, answer: str) -> RawVerdict:
        response = cast(
            dict[str, Any],
            structured.invoke(
                [
                    ("system", JUDGE_SYSTEM_PROMPT),
                    (
                        "human",
                        f"QUESTION:\n{question}\n\nGROUND TRUTH:\n{ground_truth}\n\n"
                        f"SYSTEM ANSWER:\n{answer}",
                    ),
                ]
            ),
        )
        parsed = response.get("parsed")
        if not isinstance(parsed, JudgeVerdict):
            raise ValueError(
                f"the judge returned no usable verdict: {response.get('parsing_error')}"
            )
        prompt_tokens, completion_tokens = _usage(response)
        return RawVerdict(
            verdict=parsed, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )

    return verdict_of


def build_context_assessor(config: Config) -> ContextAssessor:
    """The live retrieval-sufficiency assessor, used only with --assess-context."""
    structured = _structured(config, ContextVerdict)
    budget = config.eval.context_char_budget

    def assess(question: str, ground_truth: str, context: str) -> str:
        if not context.strip():
            return "absent"
        trimmed = (
            context if len(context) <= budget else context[:budget] + "\n\n[context truncated]"
        )
        response = cast(
            dict[str, Any],
            structured.invoke(
                [
                    ("system", CONTEXT_SYSTEM_PROMPT),
                    (
                        "human",
                        f"QUESTION:\n{question}\n\nGROUND TRUTH:\n{ground_truth}\n\n"
                        f"CONTEXT:\n{trimmed}",
                    ),
                ]
            ),
        )
        parsed = response.get("parsed")
        if not isinstance(parsed, ContextVerdict):
            raise ValueError("the context assessor returned no usable verdict")
        return parsed.sufficiency

    return assess


class Coverage(BaseModel):
    """How many of a question's expected points the answer reached."""

    total: int = 0
    covered: int = 0
    partial: int = 0

    # Computed so `--json` carries the fraction rather than making every consumer
    # re-derive it from the three counts.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def score(self) -> float | None:
        """Covered points as a fraction, half credit for partial ones.

        `None` when the question lists no points, which is not the same as zero
        coverage and must not be averaged in as if it were.
        """
        if self.total <= 0:
            return None
        return (self.covered + 0.5 * self.partial) / self.total


def _tally_points(verdict: JudgeVerdict, total: int) -> Coverage:
    """Turn the judge's point indices into a coverage count.

    Out-of-range indices are dropped and a point named both covered and partial
    counts once, as covered: the judge is a language model, and a hallucinated
    index must not push coverage above 100%.
    """
    covered = {index for index in verdict.covered_points if 1 <= index <= total}
    partial = {index for index in verdict.partial_points if 1 <= index <= total} - covered
    return Coverage(total=total, covered=len(covered), partial=len(partial))


class Judgement(BaseModel):
    """The graded outcome of one answer."""

    grade: Grade
    rationale: str = ""
    coverage: Coverage = Field(default_factory=Coverage)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def score(self) -> int:
        return SCORES[self.grade]


def judge_answer(
    question: EvalQuestion,
    answer: str,
    verdict_of: VerdictFn,
    attempts: int = 3,
) -> Judgement:
    """Grade one answer, retrying a failed grading call.

    Never raises. A judge that fails after `attempts` yields `Grade.ERROR`, which
    scores zero but is counted apart from a genuine `wrong` — losing the whole run
    to one flaky structured-output response would be the worse outcome (§10).
    """
    total = len(question.expected_points)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = verdict_of(question.question, question.ground_truth, answer)
        except Exception as exc:  # noqa: BLE001 - the provider raises assorted errors
            last_error = exc
            logger.debug(
                "judge attempt %d/%d failed for %s: %s", attempt, attempts, question.id, exc
            )
            continue
        return Judgement(
            grade=Grade(raw.verdict.grade),
            rationale=raw.verdict.rationale,
            coverage=_tally_points(raw.verdict, total),
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
        )

    logger.warning("could not grade %s after %d attempts: %s", question.id, attempts, last_error)
    return Judgement(
        grade=Grade.ERROR,
        rationale=f"The judge could not be reached: {last_error}",
        coverage=Coverage(total=total),
    )


# --- running the set -----------------------------------------------------


class QuestionOutcome(BaseModel):
    """One graded row of the report."""

    id: str
    question: str
    grade: Grade
    score: int
    rationale: str = ""
    coverage: Coverage = Field(default_factory=Coverage)
    answer: str = ""
    #: Retrieval, independent of what the answer did with it.
    chunks_retrieved: int = 0
    hit: bool | None = None
    cited_expected: bool | None = None
    cited_sources: int = 0
    context_sufficiency: str | None = None
    sources: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    judge_tokens: int = 0
    elapsed_s: float = 0.0
    #: Set when answering the question failed outright.
    error: str = ""


def _context_of(result: QueryResult) -> str:
    """The retrieved passages as one block, for the sufficiency assessment."""
    return "\n\n---\n\n".join(
        excerpt for source in result.sources for excerpt in source.excerpts if excerpt.strip()
    )


def evaluate_question(
    question: EvalQuestion,
    ask: Callable[[str], QueryResult],
    verdict_of: VerdictFn,
    assess_context: ContextAssessor | None = None,
) -> QuestionOutcome:
    """Answer one question and grade it, isolating any failure to this row.

    A question that cannot be answered is reported rather than raised: a golden
    set is run to get numbers, and losing the other twenty rows to one timeout
    would defeat the point (§10).
    """
    start = time.perf_counter()
    try:
        result = ask(question.question)
    except ZoteroRagError as exc:
        logger.warning("could not answer %s: %s", question.id, exc)
        return QuestionOutcome(
            id=question.id,
            question=question.question,
            grade=Grade.ERROR,
            score=SCORES[Grade.ERROR],
            coverage=Coverage(total=len(question.expected_points)),
            elapsed_s=time.perf_counter() - start,
            error=str(exc),
        )

    # Nothing retrieved means no answer was possible, so there is nothing for the
    # judge to weigh and no reason to spend a call finding that out.
    if not result.sources:
        judgement = Judgement(
            grade=Grade.NO_ANSWER,
            rationale="Nothing was retrieved for this question.",
            coverage=Coverage(total=len(question.expected_points)),
        )
    else:
        judgement = judge_answer(question, result.answer, verdict_of)

    sufficiency: str | None = None
    if assess_context is not None:
        try:
            sufficiency = assess_context(
                question.question, question.ground_truth, _context_of(result)
            )
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail a run
            logger.warning("context assessment failed for %s: %s", question.id, exc)

    return QuestionOutcome(
        id=question.id,
        question=question.question,
        grade=judgement.grade,
        score=judgement.score,
        rationale=judgement.rationale,
        coverage=judgement.coverage,
        answer=result.answer,
        chunks_retrieved=result.chunks_retrieved,
        hit=retrieval_hit(result.sources, question),
        cited_expected=cited_expected(result.sources, question),
        cited_sources=sum(1 for source in result.sources if source.cited),
        context_sufficiency=sufficiency,
        sources=[source.label for source in result.sources],
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        embedding_tokens=result.usage.embedding_tokens,
        judge_tokens=judgement.prompt_tokens + judgement.completion_tokens,
        elapsed_s=time.perf_counter() - start,
    )


def evaluate(
    questions: Iterable[EvalQuestion],
    ask: Callable[[str], QueryResult],
    verdict_of: VerdictFn,
    assess_context: ContextAssessor | None = None,
    on_start: Callable[[EvalQuestion], None] | None = None,
) -> list[QuestionOutcome]:
    """Run and grade every question, in file order."""
    outcomes: list[QuestionOutcome] = []
    for question in questions:
        if on_start is not None:
            on_start(question)
        outcomes.append(evaluate_question(question, ask, verdict_of, assess_context))
    return outcomes


# --- the report ----------------------------------------------------------


def _rate(values: list[bool]) -> float | None:
    """Fraction of `True` among the questions that declared an expectation."""
    return sum(values) / len(values) if values else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


class EvalAggregate(BaseModel):
    """Query-time metrics across the whole set."""

    questions: int = 0
    mean_score: float = 0.0
    grade_counts: dict[str, int] = Field(default_factory=dict)
    #: Over the questions that name an expected item; `None` when none do.
    hit_rate: float | None = None
    citation_rate: float | None = None
    #: Over the questions that list expected points.
    mean_coverage: float | None = None
    graded_questions: int = 0
    context_counts: dict[str, int] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    judge_tokens: int = 0
    elapsed_s: float = 0.0
    errors: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return (
            self.prompt_tokens + self.completion_tokens + self.embedding_tokens + self.judge_tokens
        )


def aggregate(outcomes: Sequence[QuestionOutcome]) -> EvalAggregate:
    """Summarise the rows. Metrics with no applicable question stay `None`."""
    if not outcomes:
        return EvalAggregate()

    grade_counts = {grade.value: 0 for grade in Grade}
    for outcome in outcomes:
        grade_counts[outcome.grade.value] += 1

    context_counts = {label: 0 for label in CONTEXT_LABELS}
    for outcome in outcomes:
        if outcome.context_sufficiency in context_counts:
            context_counts[outcome.context_sufficiency] += 1

    coverages = [score for outcome in outcomes if (score := outcome.coverage.score) is not None]
    return EvalAggregate(
        questions=len(outcomes),
        mean_score=sum(outcome.score for outcome in outcomes) / len(outcomes),
        grade_counts={label: count for label, count in grade_counts.items() if count},
        hit_rate=_rate([o.hit for o in outcomes if o.hit is not None]),
        citation_rate=_rate([o.cited_expected for o in outcomes if o.cited_expected is not None]),
        mean_coverage=_mean(coverages),
        graded_questions=len(coverages),
        context_counts={label: count for label, count in context_counts.items() if count},
        prompt_tokens=sum(o.prompt_tokens for o in outcomes),
        completion_tokens=sum(o.completion_tokens for o in outcomes),
        embedding_tokens=sum(o.embedding_tokens for o in outcomes),
        judge_tokens=sum(o.judge_tokens for o in outcomes),
        elapsed_s=sum(o.elapsed_s for o in outcomes),
        errors=sum(1 for o in outcomes if o.grade is Grade.ERROR),
    )


class RunSettings(BaseModel):
    """Everything that affects the numbers, recorded so two runs compare honestly.

    Without this a report is a set of figures with no statement of what produced
    them, and the first question anyone asks of a retrieval change — "compared to
    what?" — has no answer.
    """

    collection_name: str | None = None
    collection_key: str | None = None
    llm_model: str = ""
    embedding_model: str = ""
    judge_model: str = ""
    top_k: int = 0
    chunk_size: int = 0
    chunk_overlap: int = 0
    temperature: float = 0.0
    judge_temperature: float = 0.0
    #: Vector retrieval only until M7 adds alternatives; recorded now so a later
    #: report cannot be mistaken for one produced by this path.
    retrieval: str = "vector"
    reranker: str | None = None
    questions_file: str = ""
    store_chunks: int = 0
    embedding_dim: int | None = None


def run_settings(
    config: Config, store: VectorStore, top_k: int, questions_path: Path
) -> RunSettings:
    """Snapshot the settings behind a run."""
    return RunSettings(
        collection_name=config.index.collection_name,
        collection_key=config.index.collection_key,
        llm_model=config.llm_model,
        embedding_model=config.embedding_model,
        judge_model=config.judge_model,
        top_k=top_k,
        chunk_size=config.index.chunk_size,
        chunk_overlap=config.index.chunk_overlap,
        temperature=config.index.temperature,
        judge_temperature=config.eval.judge_temperature,
        reranker=config.global_.defaults.reranker,
        questions_file=str(questions_path),
        store_chunks=store.count(),
        embedding_dim=store.derived_dimension(),
    )


class EvalReport(BaseModel):
    """The whole run: what produced it, what it scored, and every row."""

    settings: RunSettings
    aggregate: EvalAggregate
    results: list[QuestionOutcome] = Field(default_factory=list)
