"""Write a starting `questions.yaml` from the documents already indexed (§11.3.2).

M6 made evaluation executable but left the golden set to be written by hand, and
the two sets that exist show how much that costs: 21 questions took an afternoon,
and the 3-question `Writing` set turned out too small to resolve any tuning
decision (§11.3.1). A collection is only measurable once someone has written
enough questions about it, so the harness is bounded by that effort.

**Questions are generated from the vector store, not from `cache/`.** The cache is
the more obvious source — the files are right there — but the store is the correct
one. A question written from text that is *not* in the index can never be answered
from the index, so it would sit in the golden set as a permanent false failure and
quietly depress every score. Generating from the store guarantees the opposite:
every question is answerable in principle, so a failure means retrieval or
generation genuinely failed. It also means no downloads, no re-parsing, and it
still works after `index --force --refetch` has emptied the cache.

The other property worth having is free here: the generator knows which item each
document came from, so `expected_items` is filled in exactly. That is what made the
`Networking` set a real discrimination test while the hand-written `Writing` set
could only check that retrieval stayed inside the collection (§12.1, §12.2).

**What a generated set is not.** These questions are derived from single documents,
so they test whether retrieval can find the passage a question was written from.
That is a genuine and useful regression harness, and it is not the same as asking
what you actually want to know: a generated set under-tests synthesis across
several papers, and it cannot ask a question the corpus fails to answer. Treat the
output as a starting point to edit, not as a finished golden set.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import Any, cast

import yaml
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from zotero_rag.config import Config
from zotero_rag.embedding import count_tokens
from zotero_rag.evaluation import EvalQuestion, QuestionSetError, evidence_found, load_questions
from zotero_rag.llm import usage_from_message
from zotero_rag.vectorstore import StoredChunk, VectorStore, reading_order_key

logger = logging.getLogger(__name__)

#: Where the sampled passages are joined when a document exceeds the budget.
ELLIPSIS = "\n\n[...]\n\n"

#: Evenly spaced passages a long document is sampled into. Head-and-tail alone
#: showed the generator only the abstract, introduction and conclusions, so every
#: question it wrote was "what problem / approach / results"; the method and the
#: experiments in the middle are where the specific, checkable details live.
SAMPLE_PASSAGES = 5


class GenerationError(QuestionSetError):
    """A question set could not be generated or written."""


# --- assembling each document from its chunks ----------------------------


class DocumentText(BaseModel):
    """One indexed item's text, reassembled from the chunks in the store."""

    item_key: str
    title: str = "(untitled)"
    authors: str = ""
    year: str = ""
    text: str = ""
    chunks: int = 0
    #: True when the document was longer than the budget and was sampled.
    truncated: bool = False

    @property
    def label(self) -> str:
        head = self.title
        if self.authors:
            head = f"{head} — {self.authors}"
        if self.year:
            head = f"{head} ({self.year})"
        return head

    @property
    def prompt_tokens_estimate(self) -> int:
        """Roughly what showing this document to the generator will cost."""
        return count_tokens([self.text])


def sample_text(text: str, budget: int) -> tuple[str, bool]:
    """Fit `text` into `budget` characters as evenly spaced passages.

    The first passage starts at the beginning (the abstract) and the last ends at
    the end (the conclusions); the rest are spread across the middle.
    """
    if len(text) <= budget:
        return text, False
    passages = max(1, min(SAMPLE_PASSAGES, budget // 1000))
    width = budget // passages
    if passages == 1:
        return text[:width], True
    step = (len(text) - width) / (passages - 1)
    starts = [round(index * step) for index in range(passages)]
    return ELLIPSIS.join(text[start : start + width] for start in starts), True


def collect_documents(store: VectorStore, budget: int) -> list[DocumentText]:
    """Reassemble every indexed item's text, one `DocumentText` per Zotero item.

    Grouped by *item*, not by attachment: a paper and its supplement are one
    document to ask questions about, and `expected_items` is item-keyed anyway.

    Adjacent chunks overlap by `chunk_overlap` characters, so the joined text
    repeats a little at each boundary. That is harmless for generation and cheaper
    than trying to guess where the splitter actually cut.
    """
    grouped: dict[str, list[StoredChunk]] = {}
    for chunk in store.all_chunks():
        item_key = str(chunk.metadata.get("item_key") or "")
        if not item_key:
            # Pre-M4 chunks may lack it; without an item key there is nothing to
            # cite, so such a document cannot carry an expectation.
            continue
        grouped.setdefault(item_key, []).append(chunk)

    documents: list[DocumentText] = []
    for item_key, chunks in grouped.items():
        chunks.sort(key=reading_order_key)
        joined = "\n".join(chunk.text.strip() for chunk in chunks if chunk.text.strip())
        text, truncated = sample_text(joined, budget)
        first = chunks[0].metadata
        documents.append(
            DocumentText(
                item_key=item_key,
                title=str(first.get("title") or "(untitled)"),
                authors=str(first.get("authors") or ""),
                year=str(first.get("year") or ""),
                text=text,
                chunks=len(chunks),
                truncated=truncated,
            )
        )

    # Stable order, so two runs over one store produce the same file.
    documents.sort(key=lambda document: (document.title.casefold(), document.item_key))
    return documents


# --- the generator -------------------------------------------------------


#: The kinds of question the generator spreads a document's questions across. The
#: first version asked for "the main points", and every document came back as the
#: same three: what problem, what approach, what results. Those test little beyond
#: whether retrieval finds the abstract.
QUESTION_KINDS = {
    "detail": "a specific fact: a number, a dataset, a parameter, a named component",
    "mechanism": "how one particular step or component works",
    "rationale": "why a design choice was made, or what it is meant to avoid",
    "comparison": "how the work compares with a baseline or alternative it discusses",
    "limitation": "a weakness, failure case or open problem the text states",
    "condition": "when or under what circumstances something holds, helps or fails",
}


class GeneratedQuestion(BaseModel):
    """One question the model wrote about a document."""

    kind: str = Field(
        default="",
        description="Which kind of question this is: " + ", ".join(QUESTION_KINDS) + ".",
    )
    question: str = Field(description="The question, answerable from this document alone.")
    evidence: str = Field(
        default="",
        description="The one or two sentences of the document that answer the "
        "question, copied exactly, character for character: no paraphrase, no "
        "ellipsis, nothing added.",
    )
    answer: str = Field(
        description="A complete reference answer, drawn only from the document text."
    )
    expected_points: list[str] = Field(
        default_factory=list,
        description="The distinct checkable claims the reference answer makes, one "
        "short sentence each, 3 to 7 of them. Each must be supported by the "
        "document; do not add anything the answer does not state.",
    )


class GeneratedSet(BaseModel):
    """Structured output schema for one document's questions."""

    questions: list[GeneratedQuestion] = Field(default_factory=list)


_KIND_LINES = "\n".join(f"- {kind}: {meaning}" for kind, meaning in QUESTION_KINDS.items())

GENERATOR_SYSTEM_PROMPT = f"""You write evaluation questions for a \
retrieval-augmented search system over a library of academic documents. The system \
must find the right passages among many papers on related subjects, so a good \
question is one a researcher would really ask, that this document answers better \
than any of its neighbours, and that does not hand the search engine the answer.

Given one document, write questions alongside a reference answer for each and the \
list of distinct claims that answer makes. Spread the questions across these kinds:
{_KIND_LINES}

Rules:
- Ask ONE thing per question, in at most 25 words, the way a researcher would \
type it into a search box. No "and how / and why / and what" second halves.
- Ask in your own words. Do not copy distinctive phrases, identifiers, \
equation symbols or numbers from the text into the question: the system has to \
find the passage from what the question means, not from an exact string it \
contains. The answer may of course be a number; the question must not give it.
- For each question, copy into `evidence` the one or two sentences of the text \
that answer it, exactly as they appear: same words, same order, no ellipsis. \
They are matched against the text automatically, so a paraphrase is useless.
- Use ONLY the supplied document text. Never add outside knowledge, and never \
invent a result, a number, or a citation that is not there.
- Draw on the whole text, not just the opening: the method and the experiments \
are where the specific, checkable details are.
- Do NOT write generic questions that fit any paper: no "what problem does the \
research address", "what approach does the study propose", "what are the main \
results or conclusions", "what is this paper about".
- Do NOT name the document's title or any part of it, its authors, or the name \
the document coins for its own method, system, model, framework or dataset. \
Describe the idea instead ("a method that has the LLM iteratively explore paths \
in a knowledge graph" rather than its name). Established names that the document \
did not coin (BERT, Wikidata, RAG) are fine.
- Each question must still stand on its own and point to this document's \
content: never "the proposed method", "the study", "this paper", "its \
experiments" or "the benchmark datasets" without saying what they are, and never \
"according to the document" or "as the authors state". A reader who has not seen \
the document must know what is being asked.
- Do NOT ask about the document as an artefact — no questions about page numbers, \
section titles, figure or table numbers, the reference list, or formatting.
- The expected points must be independently checkable and must each be stated by \
your own reference answer. Between 3 and 7 per question.
- If the supplied text is too short or too fragmentary to support a real question, \
return fewer questions, or none at all, rather than inventing content.

Long documents are sampled as several passages from beginning to end, joined by \
[...]. Do not ask about anything that would have been in an omitted part.

Bad: "What results does the study present regarding the performance of its \
proposed method?" (generic, and names nothing)
Bad: "How does Think-on-Graph improve LLM reasoning?" (the document's own name \
for its method makes the search trivial)
Bad: "In the ablation with GPT3.5-turbo, how does accuracy change as entities per \
concept group go from 0 to 3, and how large is the sparse graph used?" (two \
questions, and it copies the passage's own terms)
Good: "How deep does an LLM need to search a knowledge graph before its answers \
stop improving?"
"""


#: The generation network seam: (document, count, language) -> questions + usage.
GeneratorFn = Callable[[DocumentText, int, str], "RawGeneration"]

#: Default for `--language`: let each document set the language of its questions.
MATCH_DOCUMENT = "the same language as the document"


class RawGeneration(BaseModel):
    """What one generation call produced and what it cost."""

    questions: list[GeneratedQuestion] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def build_generator(config: Config) -> GeneratorFn:
    """The live generator, calling the configured question-writing model."""
    config.global_.require_openrouter()
    chat = ChatOpenAI(
        model=config.generator_model,
        # Questions are written once and then edited by hand; determinism matters
        # more than variety, and this keeps a re-run from churning the whole file.
        temperature=config.index.temperature,
        openai_api_base=config.global_.endpoint,
        openai_api_key=SecretStr(config.global_.api_key),
    )
    structured = chat.with_structured_output(GeneratedSet, include_raw=True)

    def generate_for(document: DocumentText, count: int, language: str) -> RawGeneration:
        response = cast(
            dict[str, Any],
            structured.invoke(
                [
                    ("system", GENERATOR_SYSTEM_PROMPT),
                    (
                        "human",
                        f"Write exactly {count} question(s) about the following "
                        f"document, spread across as many kinds as you can. Write the "
                        f"questions, the reference answers and the expected points in "
                        f"{language}.\n\n"
                        f"So that you can avoid them, the document's title and authors "
                        f"are below. Never use them, or a name coined in the title, in "
                        f"a question.\n"
                        f"TITLE: {document.title}\n"
                        f"AUTHORS: {document.authors or '(unknown)'}\n\n"
                        f"DOCUMENT TEXT:\n{document.text}",
                    ),
                ]
            ),
        )
        parsed = response.get("parsed")
        if not isinstance(parsed, GeneratedSet):
            raise GenerationError(
                f"the generator returned no usable questions: {response.get('parsing_error')}"
            )
        raw = response.get("raw")
        prompt_tokens, completion_tokens = (
            usage_from_message(raw) if isinstance(raw, AIMessage) else (0, 0)
        )
        return RawGeneration(
            questions=parsed.questions,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    return generate_for


# --- turning generated questions into a golden set -----------------------

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")

#: Longest title-derived stem, before trimming back to a word boundary.
_TITLE_STEM_CHARS = 32


def _slug(text: str) -> str:
    """A lowercase ASCII slug, folding accents rather than deleting them.

    Decomposing first matters: without it "Fröhlich" loses its `ö` to the strip
    and becomes `fr-hlich`, which is a poor id to have to type into `--only`.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _SLUG_STRIP.sub("-", unaccented.casefold()).strip("-")


def document_stem(document: DocumentText) -> str:
    """A readable id prefix: first author's surname and the year, else the title."""
    surname = document.authors.split(";")[0].split(",")[0].strip()
    parts = [part for part in (_slug(surname), _slug(document.year)) if part]
    if not parts:
        parts = [_title_stem(document.title) or _slug(document.item_key)]
    return "-".join(parts)


def _title_stem(title: str) -> str:
    """A title-derived stem, cut at a word boundary rather than mid-word."""
    slug = _slug(title)
    if len(slug) <= _TITLE_STEM_CHARS:
        return slug
    clipped = slug[:_TITLE_STEM_CHARS]
    # Prefer the last whole word, unless that leaves almost nothing.
    boundary = clipped.rfind("-")
    return clipped[:boundary] if boundary >= _TITLE_STEM_CHARS // 2 else clipped


def assign_ids(stem: str, count: int, taken: set[str]) -> list[str]:
    """`stem-1 … stem-n`, suffixed further if a stem repeats across documents.

    Ids key the report and `load_questions` rejects duplicates, so two papers by
    the same author in the same year must not collide.
    """
    prefix, suffix = stem, 2
    while any(f"{prefix}-{number}" in taken for number in range(1, count + 1)):
        prefix, suffix = f"{stem}-{suffix}", suffix + 1
    ids = [f"{prefix}-{number}" for number in range(1, count + 1)]
    taken.update(ids)
    return ids


def to_questions(
    document: DocumentText, generated: Sequence[GeneratedQuestion], taken: set[str]
) -> list[EvalQuestion]:
    """Convert one document's generated questions into golden-set entries."""
    usable = [
        question
        for question in generated
        if question.question.strip() and (question.answer.strip() or question.expected_points)
    ]
    ids = assign_ids(document_stem(document), len(usable), taken)
    return [
        EvalQuestion(
            id=identifier,
            question=question.question.strip(),
            answer=question.answer.strip(),
            expected_points=[point.strip() for point in question.expected_points if point.strip()],
            # The whole point: provenance is known exactly, not guessed.
            expected_items=[document.item_key],
            expected_evidence=[question.evidence.strip()] if question.evidence.strip() else [],
        )
        for identifier, question in zip(ids, usable, strict=True)
    ]


#: Words too common to make a run of title words a giveaway.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "over",
        "the",
        "to",
        "towards",
        "under",
        "via",
        "what",
        "when",
        "with",
        "without",
    }
)

#: Consecutive content words of the title that count as quoting it. Shorter runs
#: ("knowledge graph completion") are the subject, which a question has to name;
#: so is the whole of a title shorter than `_MIN_TITLE_RUN` ("Knowledge Graphs").
_TITLE_RUN = 5
_MIN_TITLE_RUN = 3

#: Longest title prefix (before a colon) treated as a coined name: "GIVE",
#: "Think-on-Graph", "WikiGraphs".
_COINED_NAME_WORDS = 3

_WORD = re.compile(r"[\w-]+")

#: A question that points at "the document" or "the proposed method" only makes
#: sense to someone already holding it, and gives retrieval nothing to search for.
_SELF_REFERENCE = re.compile(
    r"\b(?:(?:the|this) (?:document|paper|study|article|text|work|research)\b"
    r"|the authors\b|(?:the|this|its) proposed\b)",
    re.IGNORECASE,
)

#: Questions asked for beyond the ones wanted, so that dropping a giveaway does
#: not leave a document short.
SPARE_QUESTIONS = 2


def _content_words(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.casefold()) if word not in _STOPWORDS]


def _coined_name(title: str) -> str | None:
    """The short name before a title's colon, which papers use for their method."""
    head, colon, _ = title.partition(":")
    head = head.strip()
    if not colon or not head or len(head.split()) > _COINED_NAME_WORDS:
        return None
    return head


def _surnames(authors: str) -> list[str]:
    """Surnames from a `Last, First; Last, First` author string."""
    return [name for entry in authors.split(";") if (name := entry.split(",")[0].strip())]


def _contains_word(text: str, word: str) -> bool:
    """Whole-word, case-sensitive: "GIVE" is a name, "give" is a verb; "He" is a
    surname, "he" a pronoun."""
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text) is not None


def giveaway(document: DocumentText, question: str) -> str | None:
    """Why `question` hands retrieval its document, or `None` if it does not.

    The prompt forbids these, but a rule a model only usually follows is not
    one a golden set can rest on: a question that names its paper measures a
    keyword match, not retrieval, and one that says "the document" names nothing
    for retrieval to find.
    """
    if match := _SELF_REFERENCE.search(question):
        return f"refers to {match.group(0)!r} without saying what it is"
    for surname in _surnames(document.authors):
        if _contains_word(question, surname):
            return f"names the author {surname!r}"
    if (name := _coined_name(document.title)) and _contains_word(question, name):
        return f"names the document's own {name!r}"

    title = _content_words(document.title)
    run = min(_TITLE_RUN, len(title))
    if run >= _MIN_TITLE_RUN:
        asked = " ".join(_content_words(question))
        for start in range(len(title) - run + 1):
            if " ".join(title[start : start + run]) in asked:
                return "quotes the title"
    return None


class GenerationStats(BaseModel):
    """What a generation run did, for the summary line."""

    documents: int = 0
    questions: int = 0
    truncated: int = 0
    #: Questions dropped for naming their document; see `giveaway`.
    rejected: int = 0
    #: Questions kept without an evidence quote, so without a passage check.
    without_evidence: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    failures: list[str] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def generate_questions(
    documents: Iterable[DocumentText],
    generate_for: GeneratorFn,
    per_document: int,
    language: str = MATCH_DOCUMENT,
    on_start: Callable[[DocumentText], None] | None = None,
) -> tuple[list[EvalQuestion], GenerationStats]:
    """Write questions for every document, isolating a failure to one document.

    A document the generator chokes on is reported and skipped: the run's value is
    the questions it did produce, and losing fifteen documents to one bad response
    would be the worse outcome (§10).
    """
    start = time.perf_counter()
    stats = GenerationStats()
    questions: list[EvalQuestion] = []
    taken: set[str] = set()

    for document in documents:
        if on_start is not None:
            on_start(document)
        stats.documents += 1
        if document.truncated:
            stats.truncated += 1
        try:
            raw = generate_for(document, per_document + SPARE_QUESTIONS, language)
        except Exception as exc:  # noqa: BLE001 - the provider raises assorted errors
            logger.warning("could not generate questions for %s: %s", document.label, exc)
            stats.failures.append(f"{document.label}: {exc}")
            continue
        stats.prompt_tokens += raw.prompt_tokens
        stats.completion_tokens += raw.completion_tokens
        kept: list[GeneratedQuestion] = []
        for generated in raw.questions:
            if reason := giveaway(document, generated.question):
                logger.info("dropped a question that %s: %s", reason, generated.question)
                stats.rejected += 1
                continue
            if generated.evidence and not evidence_found(generated.evidence, [document.text]):
                # A paraphrase cannot be matched against retrieved chunks, so it
                # would score as a miss for every chunking alike. The question is
                # still good; only its passage check is lost.
                logger.info("evidence not found verbatim, dropped it: %s", generated.evidence)
                generated = generated.model_copy(update={"evidence": ""})
            kept.append(generated)
        kept = kept[:per_document]
        stats.without_evidence += sum(1 for generated in kept if not generated.evidence)
        produced = to_questions(document, kept, taken)
        if not produced:
            logger.warning("the generator produced nothing usable for %s", document.label)
            stats.failures.append(f"{document.label}: no usable question was returned")
        questions.extend(produced)

    stats.questions = len(questions)
    stats.elapsed_s = time.perf_counter() - start
    return questions, stats


# --- writing the file ----------------------------------------------------


class _BlockDumper(yaml.SafeDumper):
    """A dumper that renders multi-line strings as literal blocks."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Readable output without hand-quoting.

    Reference answers are prose full of colons, quotes and accented text, all of
    which are YAML hazards in a plain scalar. Letting PyYAML decide the quoting is
    what makes the emitted file *always* parseable; the literal-block style is
    only for readability, and it is safe here because trailing whitespace — which
    a literal block cannot represent — has been stripped from every line.
    """
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockDumper.add_representer(str, _represent_str)


def _clean(text: str) -> str:
    """Normalise line endings and drop trailing spaces, so `|` blocks are legal."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def dump_questions(
    questions: Sequence[EvalQuestion], header: str = "", version_id: str | None = None
) -> str:
    """Serialise a golden set to YAML in the §11.3 schema."""
    payload: dict[str, Any] = {} if version_id is None else {"version_id": version_id}
    payload["questions"] = [
        {
            "id": question.id,
            "question": _clean(question.question),
            "expected_items": list(question.expected_items),
            **(
                {"expected_evidence": [_clean(quote) for quote in question.expected_evidence]}
                if question.expected_evidence
                else {}
            ),
            "expected_points": [_clean(point) for point in question.expected_points],
            "answer": _clean(question.answer),
        }
        for question in questions
    ]
    body = yaml.dump(
        payload,
        Dumper=_BlockDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=88,
        indent=4,
    )
    return f"{header}\n{body}" if header else body


def default_version_id(config: Config, now: datetime | None = None) -> str:
    """`kg-20261031-1105`: the collection and when the set was generated."""
    stem = _slug(config.index.collection_name or "") or "questions"
    return f"{stem}-{(now or datetime.now()):%Y%m%d-%H%M}"


def file_header(config: Config, stats: GenerationStats, language: str) -> str:
    """A comment block saying what produced this file and what to do with it."""
    collection = config.index.collection_name or "(unpinned)"
    return "\n".join(
        [
            f"# Question set generated from the `{collection}` collection by",
            f"# `zotero-rag generate-questions` using {config.generator_model}.",
            f"# {stats.questions} question(s) from {stats.documents} indexed document(s),"
            f" {language}.",
            "#",
            "# EDIT THIS FILE. It is a starting point, not a finished golden set:",
            "#   - every question was written from a single document, so the set tests",
            "#     whether retrieval finds that document, not synthesis across several;",
            "#   - a generated set cannot ask something the corpus fails to answer, so",
            "#     add your own questions for the things you actually want to know;",
            "#   - `expected_items` is exact — it is the item the text came from — but",
            "#     the questions and points are a model's reading and deserve a review.",
            "#",
            "# Runs are logged in reports/experiments-questions-<version_id>.json. Once",
            "# runs are logged, change `version_id` whenever you edit the questions, so",
            "# results on the old and the new set land in separate logs.",
            "#",
            "# Then: zotero-rag eval",
        ]
    )


def write_questions(
    path: Any,
    questions: Sequence[EvalQuestion],
    header: str = "",
    force: bool = False,
    version_id: str | None = None,
) -> None:
    """Write the set to `path`, refusing to clobber an existing file.

    The written file is read back with `load_questions` before the call returns, so
    `generate-questions` can never emit something `eval` cannot parse.
    """
    if path.exists() and not force:
        raise GenerationError(
            f"{path} already exists. Generated questions would overwrite a set you may "
            "have edited; pass --force to replace it, or --out to write elsewhere."
        )
    text = dump_questions(questions, header, version_id)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise GenerationError(f"Cannot write {path}: {exc}") from exc

    try:
        reloaded = load_questions(path)
    except QuestionSetError as exc:  # pragma: no cover - a dumper bug, not user input
        raise GenerationError(
            f"The generated set was written to {path} but does not parse: {exc}"
        ) from exc
    if len(reloaded) != len(questions):  # pragma: no cover - as above
        raise GenerationError(
            f"The generated set at {path} reloaded as {len(reloaded)} of "
            f"{len(questions)} questions."
        )
