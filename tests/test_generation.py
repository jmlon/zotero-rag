"""Generating a question set from an indexed collection (§11.3.2).

The store is a real on-disk Chroma instance; only the generation call is a local
callable. The round trip that matters most is asserted directly: whatever the
generator writes must be loadable by `eval`'s own parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zotero_rag.config import Config
from zotero_rag_core.evaluation import EvalQuestion, load_questions
from zotero_rag_core.generation import (
    ELLIPSIS,
    MATCH_DOCUMENT,
    SPARE_QUESTIONS,
    DocumentText,
    GeneratedQuestion,
    GenerationError,
    GenerationStats,
    RawGeneration,
    assign_ids,
    collect_documents,
    document_stem,
    dump_questions,
    file_header,
    generate_questions,
    giveaway,
    sample_text,
    to_questions,
    write_questions,
)
from zotero_rag_core.vectorstore import VectorStore


def store_with(tmp_path: Path, chunks: list[dict[str, object]]) -> VectorStore:
    """A store holding the given chunks; `text` and metadata per entry."""
    store = VectorStore(tmp_path / "chroma")
    store.add(
        ids=[str(chunk["id"]) for chunk in chunks],
        embeddings=[[1.0, 0.0, 0.5] for _ in chunks],
        documents=[str(chunk["text"]) for chunk in chunks],
        metadatas=[
            {key: value for key, value in chunk.items() if key not in {"id", "text"}}
            for chunk in chunks
        ],
    )
    return store


def chunk(
    identifier: str,
    text: str,
    item_key: str = "ITEM1",
    attachment_key: str = "ATT1",
    page: int = 1,
    chunk_index: int = 0,
    title: str = "A Paper About Fog",
    authors: str = "Tordera, Eva; Rovira, Joan",
    year: str = "2016",
) -> dict[str, object]:
    return {
        "id": identifier,
        "text": text,
        "item_key": item_key,
        "attachment_key": attachment_key,
        "page": page,
        "chunk_index": chunk_index,
        "title": title,
        "authors": authors,
        "year": year,
    }


# --- reassembling documents from the store -------------------------------


def test_a_document_is_reassembled_from_its_chunks(tmp_path: Path) -> None:
    store = store_with(
        tmp_path,
        [
            chunk("ATT1:1:0", "First page.", page=1),
            chunk("ATT1:2:0", "Second page.", page=2),
        ],
    )
    documents = collect_documents(store, budget=10_000)

    assert len(documents) == 1
    document = documents[0]
    assert document.item_key == "ITEM1"
    assert document.title == "A Paper About Fog"
    assert document.chunks == 2
    assert document.text == "First page.\nSecond page."
    assert document.truncated is False


def test_chunks_are_ordered_numerically_not_by_id(tmp_path: Path) -> None:
    """Sorting the ids as strings would put page 10 before page 2."""
    store = store_with(
        tmp_path,
        [
            chunk("ATT1:10:0", "ten", page=10),
            chunk("ATT1:2:0", "two", page=2),
            chunk("ATT1:1:1", "one-b", page=1, chunk_index=1),
            chunk("ATT1:1:0", "one-a", page=1, chunk_index=0),
        ],
    )
    assert collect_documents(store, 10_000)[0].text == "one-a\none-b\ntwo\nten"


def test_several_attachments_of_one_item_are_one_document(tmp_path: Path) -> None:
    """A paper and its supplement are one thing to ask questions about."""
    store = store_with(
        tmp_path,
        [
            chunk("ATT1:1:0", "the paper", attachment_key="ATT1"),
            chunk("ATT2:1:0", "the supplement", attachment_key="ATT2"),
        ],
    )
    documents = collect_documents(store, 10_000)
    assert len(documents) == 1
    assert documents[0].chunks == 2


def test_separate_items_become_separate_documents(tmp_path: Path) -> None:
    store = store_with(
        tmp_path,
        [
            chunk("ATT1:1:0", "first", item_key="ITEM1", title="Beta"),
            chunk("ATT2:1:0", "second", item_key="ITEM2", title="Alpha"),
        ],
    )
    documents = collect_documents(store, 10_000)
    # Sorted by title, so two runs over one store produce the same file.
    assert [document.title for document in documents] == ["Alpha", "Beta"]


def test_a_chunk_without_an_item_key_is_skipped(tmp_path: Path) -> None:
    """Without an item key nothing can be cited, so no expectation can be set."""
    store = store_with(tmp_path, [chunk("ATT1:1:0", "orphaned", item_key="")])
    assert collect_documents(store, 10_000) == []


# --- fitting a long document into the budget -----------------------------


def test_a_short_document_is_untouched() -> None:
    assert sample_text("short", 100) == ("short", False)


def test_a_long_document_is_sampled_from_beginning_to_end() -> None:
    """Head-and-tail showed only the abstract and conclusions; the middle matters."""
    text = "".join(letter * 2000 for letter in "ABCDEFGHIJ")
    sampled, truncated = sample_text(text, 5000)

    assert truncated is True
    passages = sampled.split(ELLIPSIS)
    assert len(passages) == 5
    assert all(len(passage) == 1000 for passage in passages)
    assert sampled.startswith("A")
    assert sampled.endswith("J")
    # The middle of the document is represented, not just its ends.
    assert len(set(sampled) & set("ABCDEFGHIJ")) >= 5


def test_a_tiny_budget_is_one_passage_from_the_start() -> None:
    sampled, truncated = sample_text("x" * 5000, 100)
    assert (sampled, truncated) == ("x" * 100, True)


def test_the_budget_is_applied_when_collecting(tmp_path: Path) -> None:
    store = store_with(tmp_path, [chunk("ATT1:1:0", "x" * 50000)])
    document = collect_documents(store, budget=5000)[0]
    assert document.truncated is True
    assert len(document.text) <= 5000 + 4 * len(ELLIPSIS)


# --- questions that give their document away -----------------------------

PAPER = DocumentText(
    item_key="A",
    title="Think-on-Graph: Deep and Responsible Reasoning of Large Language Model "
    "on Knowledge Graph",
    authors="Sun, Jiashuo; He, Xiaoyu",
)


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        ("How does Think-on-Graph explore relation paths?", "Think-on-Graph"),
        ("What did Sun et al. find about beam width?", "Sun"),
        ("How did He and colleagues prune the graph?", "He"),
        (
            "How does deep and responsible reasoning of large language models work?",
            "title",
        ),
    ],
)
def test_a_question_naming_its_document_is_caught(question: str, reason: str) -> None:
    found = giveaway(PAPER, question)
    assert found is not None
    assert reason in found


@pytest.mark.parametrize(
    "question",
    [
        # The subject has to be named; a few title words are not the title.
        "How can an LLM reason over a knowledge graph with beam search?",
        # Lower-case words that happen to spell a surname or a coined name.
        "What does the sun-like search give the model, and why does he matter?",
    ],
)
def test_a_question_about_the_subject_is_kept(question: str) -> None:
    assert giveaway(PAPER, question) is None


def test_a_short_title_is_the_subject_not_a_giveaway() -> None:
    document = DocumentText(item_key="A", title="Knowledge Graphs")
    assert giveaway(document, "How are knowledge graphs completed?") is None


def test_an_acronym_name_is_matched_by_case() -> None:
    document = DocumentText(item_key="A", title="GIVE: Structured Reasoning with Graphs")
    assert giveaway(document, "How does GIVE extrapolate?") is not None
    assert giveaway(document, "What does sparse evidence give the model?") is None


def test_giveaway_questions_are_dropped_and_counted() -> None:
    def generate_for(document: DocumentText, count: int, language: str) -> RawGeneration:
        return RawGeneration(
            questions=[
                GeneratedQuestion(question="How does Think-on-Graph work?", answer="x"),
                GeneratedQuestion(question="How does beam search pick relations?", answer="y"),
            ]
        )

    questions, stats = generate_questions([PAPER], generate_for, per_document=2)

    assert [question.question for question in questions] == ["How does beam search pick relations?"]
    assert stats.rejected == 1


# --- ids -----------------------------------------------------------------


def test_an_id_stem_is_the_first_author_and_year() -> None:
    document = DocumentText(
        item_key="K", title="Whatever", authors="Zhang, Junjie; Ye, Minghua", year="2020"
    )
    assert document_stem(document) == "zhang-2020"


def test_an_id_stem_falls_back_to_the_title() -> None:
    document = DocumentText(item_key="K", title="Fog Computing Survey")
    assert document_stem(document) == "fog-computing-survey"


def test_an_id_stem_falls_back_to_the_item_key_when_there_is_nothing_else() -> None:
    assert document_stem(DocumentText(item_key="ABC123", title="")) == "abc123"


def test_ids_are_numbered_within_a_document() -> None:
    assert assign_ids("zhang-2020", 3, set()) == [
        "zhang-2020-1",
        "zhang-2020-2",
        "zhang-2020-3",
    ]


def test_a_repeated_stem_is_disambiguated() -> None:
    """Two papers by one author in one year must not collide: ids key the report."""
    taken: set[str] = set()
    first = assign_ids("zhang-2020", 2, taken)
    second = assign_ids("zhang-2020", 2, taken)

    assert first == ["zhang-2020-1", "zhang-2020-2"]
    assert second == ["zhang-2020-2-1", "zhang-2020-2-2"]
    assert not set(first) & set(second)


# --- converting to golden-set entries ------------------------------------


DOCUMENT = DocumentText(
    item_key="TR6MHCJ2", title="CFR-RL", authors="Zhang, Junjie", year="2020", text="..."
)


def test_generated_questions_carry_the_item_they_came_from() -> None:
    """The whole point of generating: provenance is exact, not guessed."""
    generated = [
        GeneratedQuestion(question="What problem?", answer="This one.", expected_points=["a", "b"])
    ]
    questions = to_questions(DOCUMENT, generated, set())

    assert len(questions) == 1
    assert questions[0].id == "zhang-2020-1"
    assert questions[0].expected_items == ["TR6MHCJ2"]
    assert questions[0].expected_points == ["a", "b"]
    assert questions[0].has_expected_source


def test_an_empty_question_is_dropped_rather_than_written() -> None:
    generated = [
        GeneratedQuestion(question="   ", answer="orphan answer"),
        GeneratedQuestion(question="A real one?", answer="Yes."),
    ]
    questions = to_questions(DOCUMENT, generated, set())
    assert [question.question for question in questions] == ["A real one?"]
    # Ids stay contiguous over what was kept.
    assert questions[0].id == "zhang-2020-1"


def test_a_question_with_nothing_to_grade_against_is_dropped() -> None:
    """`load_questions` would reject it, so it must never be written."""
    generated = [GeneratedQuestion(question="Unanswerable?", answer="", expected_points=[])]
    assert to_questions(DOCUMENT, generated, set()) == []


# --- the run -------------------------------------------------------------


def generator(count: int = 2, fail_on: str = "") -> object:
    """A generation seam returning `count` questions, optionally failing on a title."""

    def generate_for(document: DocumentText, per_document: int, language: str) -> RawGeneration:
        if fail_on and document.title == fail_on:
            raise RuntimeError("the provider said no")
        return RawGeneration(
            questions=[
                GeneratedQuestion(
                    question=f"Question {number} about {document.title}?",
                    answer=f"Answer {number}.",
                    expected_points=[f"point {number}a", f"point {number}b"],
                )
                for number in range(1, count + 1)
            ],
            prompt_tokens=4000,
            completion_tokens=300,
        )

    return generate_for


def test_a_run_writes_questions_for_every_document() -> None:
    documents = [
        DocumentText(item_key="A", title="Alpha", authors="Ay, A", year="2020"),
        DocumentText(item_key="B", title="Beta", authors="Bee, B", year="2021"),
    ]
    questions, stats = generate_questions(documents, generator(2), per_document=2)  # type: ignore[arg-type]

    assert stats.documents == 2
    assert stats.questions == 4
    assert stats.prompt_tokens == 8000
    assert stats.total_tokens == 8600
    assert [question.id for question in questions] == [
        "ay-2020-1",
        "ay-2020-2",
        "bee-2021-1",
        "bee-2021-2",
    ]


def test_one_failing_document_does_not_lose_the_others() -> None:
    """PRD §10: the run's value is the questions it did produce."""
    documents = [
        DocumentText(item_key="A", title="Alpha", authors="Ay, A", year="2020"),
        DocumentText(item_key="B", title="Broken", authors="Bee, B", year="2021"),
        DocumentText(item_key="C", title="Gamma", authors="Cee, C", year="2022"),
    ]
    questions, stats = generate_questions(
        documents,
        generator(1, fail_on="Broken"),  # type: ignore[arg-type]
        per_document=1,
    )

    assert stats.questions == 2
    assert len(stats.failures) == 1
    assert "Broken" in stats.failures[0]
    assert {question.expected_items[0] for question in questions} == {"A", "C"}


def test_a_truncated_document_is_counted() -> None:
    documents = [DocumentText(item_key="A", title="Alpha", truncated=True)]
    _, stats = generate_questions(documents, generator(1), per_document=1)  # type: ignore[arg-type]
    assert stats.truncated == 1


# --- writing the file ----------------------------------------------------


def test_the_written_file_is_loadable_by_eval(tmp_path: Path) -> None:
    """The guarantee that matters: generation cannot emit what eval cannot read."""
    questions = [
        EvalQuestion(
            id="tricky-1",
            # Every YAML hazard that had to be hand-quoted in the Networking set.
            question='¿Qué significa "CFR-RL": un esquema de TE?',
            answer="Un enfoque: usa RL.\n\nY luego LP: para las rutas.\n",
            expected_points=[
                "primero: identifica flujos críticos",
                'usa "programación lineal" — con acentos',
                "  leading and trailing space  ",
            ],
            expected_items=["TR6MHCJ2"],
        )
    ]
    path = tmp_path / "questions.yaml"

    write_questions(path, questions)

    reloaded = load_questions(path)
    assert len(reloaded) == 1
    assert reloaded[0].question == '¿Qué significa "CFR-RL": un esquema de TE?'
    assert reloaded[0].expected_items == ["TR6MHCJ2"]
    assert reloaded[0].expected_points[0] == "primero: identifica flujos críticos"
    assert "leading and trailing space" in reloaded[0].expected_points[2]


def test_multi_line_answers_are_written_as_readable_blocks(tmp_path: Path) -> None:
    questions = [
        EvalQuestion(
            id="q1", question="Why?", answer="First line.\nSecond line.", expected_items=["K"]
        )
    ]
    text = dump_questions(questions)
    # A literal block, not a quoted one-liner full of \n escapes.
    assert "answer: |" in text
    assert "\\n" not in text


def test_an_existing_set_is_not_overwritten(tmp_path: Path) -> None:
    """A curated golden set is the last thing that should be clobbered."""
    path = tmp_path / "questions.yaml"
    path.write_text("questions: []\n", encoding="utf-8")
    questions = [EvalQuestion(id="q1", question="Why?", answer="Because.")]

    with pytest.raises(GenerationError, match="already exists"):
        write_questions(path, questions)

    assert path.read_text(encoding="utf-8") == "questions: []\n"


def test_force_replaces_an_existing_set(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text("questions: []\n", encoding="utf-8")

    write_questions(path, [EvalQuestion(id="q1", question="Why?", answer="Because.")], force=True)

    assert len(load_questions(path)) == 1


def test_the_header_says_the_file_is_a_draft(tmp_path: Path) -> None:
    """The output is a starting point; the file itself has to say so."""
    config = Config()
    config.index.collection_name = "Networking"
    stats = GenerationStats(documents=16, questions=48)

    header = file_header(config, stats, MATCH_DOCUMENT)

    assert "Networking" in header
    assert config.generator_model in header
    assert "EDIT THIS FILE" in header
    assert "48 question(s) from 16 indexed document(s)" in header
    # Every line is a comment, or the file will not parse.
    assert all(line.startswith("#") for line in header.splitlines() if line)


def test_a_generated_file_round_trips_through_the_header(tmp_path: Path) -> None:
    path = tmp_path / "out.yaml"
    questions = [
        EvalQuestion(id="q1", question="Why?", answer="Because.", expected_items=["K"]),
    ]
    write_questions(path, questions, header=file_header(Config(), GenerationStats(), "Spanish"))
    assert load_questions(path)[0].expected_items == ["K"]


def test_an_accented_surname_keeps_its_letters() -> None:
    """Stripping non-ASCII would make "Fröhlich" into `fr-hlich`, an awful id to type."""
    document = DocumentText(item_key="K", title="x", authors="Fröhlich, Piotr", year="2024")
    assert document_stem(document) == "frohlich-2024"


def test_a_long_title_stem_is_cut_at_a_word_boundary() -> None:
    document = DocumentText(
        item_key="K", title="Traffic engineering with reinforcement learning in SDN"
    )
    stem = document_stem(document)
    assert stem == "traffic-engineering-with"
    assert not stem.endswith("-")


@pytest.mark.parametrize(
    "question",
    [
        "What limitation does the document mention about link prediction benchmarks?",
        "How many entities does YAGO have, as stated in this paper?",
        "How does the proposed method build prompts?",
        "What do the authors conclude about fine-tuning?",
    ],
)
def test_a_question_pointing_at_its_document_is_caught(question: str) -> None:
    assert giveaway(PAPER, question) is not None


def test_spare_questions_make_up_for_dropped_ones() -> None:
    asked: list[int] = []

    def generate_for(document: DocumentText, count: int, language: str) -> RawGeneration:
        asked.append(count)
        return RawGeneration(
            questions=[
                GeneratedQuestion(question="What does the paper conclude?", answer="x"),
                GeneratedQuestion(question="How does beam search pick relations?", answer="y"),
                GeneratedQuestion(question="Why prune paths below a threshold?", answer="z"),
                GeneratedQuestion(question="When does a wider beam stop helping?", answer="w"),
            ]
        )

    questions, stats = generate_questions([PAPER], generate_for, per_document=2)

    assert asked == [2 + SPARE_QUESTIONS]
    assert [question.question for question in questions] == [
        "How does beam search pick relations?",
        "Why prune paths below a threshold?",
    ]
    assert stats.rejected == 1


def test_a_verbatim_evidence_quote_is_kept_and_a_paraphrase_is_not(tmp_path: Path) -> None:
    document = DocumentText(
        item_key="A",
        title="Paths",
        text="Setup. We set the beam width to 3 for every dataset. Results follow.",
    )

    def generate_for(document: DocumentText, count: int, language: str) -> RawGeneration:
        return RawGeneration(
            questions=[
                GeneratedQuestion(
                    question="How wide is the beam?",
                    answer="Three.",
                    evidence="We set the beam width to 3 for every dataset.",
                ),
                GeneratedQuestion(
                    question="Is the beam the same everywhere?",
                    answer="Yes.",
                    evidence="The authors always use a beam of three.",
                ),
            ]
        )

    questions, stats = generate_questions([document], generate_for, per_document=2)

    assert questions[0].expected_evidence == ["We set the beam width to 3 for every dataset."]
    assert questions[1].expected_evidence == []
    assert stats.without_evidence == 1

    path = tmp_path / "questions.yaml"
    path.write_text(dump_questions(questions))
    reloaded = load_questions(path)
    assert reloaded[0].expected_evidence == questions[0].expected_evidence
    assert reloaded[1].expected_evidence == []
