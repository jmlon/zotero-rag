"""A cumulative record of every scored run, for analysis across experiments.

`eval` and `compare-embeddings` each print a table and `compare-embeddings` saves
one to `reports/`, but those are snapshots: comparing the chunk-size sweep with
last week's reranker run means opening files side by side. This module appends
every scored configuration to a log, so a notebook can load the lot and slice it
by any setting.

**One entry per scored configuration.** An `eval` run is one entry; a
`compare-embeddings` run is one entry per variant, sharing a `batch_id`. Each
entry carries the full settings, the question set it ran on, the aggregate and
every per-question outcome, so nothing has to be joined back from elsewhere.

**One log per question set.** Scores on different question sets are not
comparable, so each set has its own file,
`reports/experiments-questions-<version_id>.json`, named by the `version_id:` at
the top of `questions.yaml` (or by the set's fingerprint when it has none). Every
entry also records a fingerprint of the set's content: a set edited without a new
`version_id` shows up as a second fingerprint in one log, and recording warns
about it rather than letting the two be pooled silently.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from zotero_rag.errors import ZoteroRagError
from zotero_rag.evaluation import EvalAggregate, QuestionOutcome, QuestionSet, RunSettings

logger = logging.getLogger(__name__)

#: `experiments-questions-<version_id>.json` inside `reports/`.
LOG_PREFIX = "experiments-questions-"

#: Bumped when an entry's shape changes in a way a notebook would trip over.
SCHEMA_VERSION = 1


class ExperimentLogError(ZoteroRagError):
    """The experiment log exists but cannot be read, so it is left untouched."""


class QuestionSetInfo(BaseModel):
    """Which questions a run answered."""

    version_id: str | None = None
    fingerprint: str
    path: str
    #: Questions in the file, and how many of them this run answered.
    questions: int
    questions_run: int
    #: The ids that ran, when `--only` narrowed the set; `None` for the whole set.
    question_ids: list[str] | None = None


class ExperimentRun(BaseModel):
    """One scored configuration."""

    run_id: str
    #: Shared by the variants of one `compare-embeddings` run; the run's own id
    #: for `eval`.
    batch_id: str
    recorded_at: str
    command: Literal["eval", "compare-embeddings"]
    question_set: QuestionSetInfo
    settings: RunSettings
    #: `compare-embeddings` only: the configured store rather than a variant.
    baseline: bool = False
    aggregate: EvalAggregate
    results: list[QuestionOutcome] = Field(default_factory=list)


class ExperimentLog(BaseModel):
    schema_version: int = SCHEMA_VERSION
    runs: list[ExperimentRun] = Field(default_factory=list)


def new_batch_id() -> str:
    return uuid.uuid4().hex[:12]


def question_set_info(question_set: QuestionSet, path: Path, ran: list[str]) -> QuestionSetInfo:
    """Describe the set a run used; `ran` is the ids of the questions it answered."""
    whole = len(ran) == len(question_set.questions)
    return QuestionSetInfo(
        version_id=question_set.version_id,
        fingerprint=question_set.fingerprint,
        path=str(path),
        questions=len(question_set.questions),
        questions_run=len(ran),
        question_ids=None if whole else ran,
    )


def make_run(
    command: Literal["eval", "compare-embeddings"],
    batch_id: str,
    question_set: QuestionSetInfo,
    settings: RunSettings,
    aggregate: EvalAggregate,
    results: list[QuestionOutcome],
    baseline: bool = False,
) -> ExperimentRun:
    return ExperimentRun(
        run_id=uuid.uuid4().hex[:12],
        batch_id=batch_id,
        recorded_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        command=command,
        question_set=question_set,
        settings=settings,
        baseline=baseline,
        aggregate=aggregate,
        results=results,
    )


def log_path(reports_dir: Path, question_set: QuestionSetInfo) -> Path:
    """The log for this question set: by `version_id`, else by fingerprint."""
    name = question_set.version_id or f"fp-{question_set.fingerprint}"
    return reports_dir / f"{LOG_PREFIX}{name}.json"


def load_log(path: Path) -> ExperimentLog:
    """The log at `path`, or an empty one if there is none yet."""
    if not path.is_file():
        return ExperimentLog()
    try:
        return ExperimentLog.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ExperimentLogError(
            f"Cannot read {path}, so this run was not added to it: {exc}. The file was "
            "left as it is; move it aside to start a new log."
        ) from exc


def version_conflicts(log: ExperimentLog, info: QuestionSetInfo) -> list[str]:
    """Fingerprints already recorded under this `version_id` that differ from `info`'s."""
    if info.version_id is None:
        return []
    return sorted(
        {
            run.question_set.fingerprint
            for run in log.runs
            if run.question_set.version_id == info.version_id
            and run.question_set.fingerprint != info.fingerprint
        }
    )


def record_runs(path: Path, runs: list[ExperimentRun]) -> list[str]:
    """Append `runs` to the log at `path`; returns warnings worth showing.

    `path` is normally `log_path(...)`, so every run in a log shares a version.

    The whole file is rewritten through a temporary file and an atomic rename, so
    an interrupted write can never leave a truncated log behind.
    """
    log = load_log(path)
    warnings: list[str] = []
    for info in {run.question_set.fingerprint: run.question_set for run in runs}.values():
        if conflicts := version_conflicts(log, info):
            warnings.append(
                f"question set {info.version_id!r} has changed since it was first recorded "
                f"(fingerprint {info.fingerprint}, earlier {', '.join(conflicts)}). Give the "
                "edited set a new `version_id:` so its runs are not mixed with the old ones."
            )
    log.runs.extend(runs)

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".experiments-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(log.model_dump_json(indent=2))
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    logger.info("recorded %d run(s) in %s", len(runs), path)
    return warnings
