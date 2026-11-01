"""Hybrid retrieval: tokenizing, BM25, and reciprocal rank fusion (§11.4.2).

The store is a real on-disk Chroma instance and BM25 is real; only the embedding
of the question is a local stand-in, as elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zotero_rag_core.querying import answer_question
from zotero_rag_core.retrieval import (
    CANDIDATE_MULTIPLIER,
    MIN_CANDIDATES,
    RRF_K,
    KeywordIndex,
    candidate_depth,
    reciprocal_rank_fusion,
    retrieve,
    tokenize,
)
from zotero_rag_core.vectorstore import StoredChunk, VectorStore


def chunk(identifier: str, text: str = "some text", distance: float | None = None) -> StoredChunk:
    return StoredChunk(
        id=identifier,
        text=text,
        metadata={"attachment_key": identifier.split(":")[0], "page": 1, "chunk_index": 0},
        distance=distance,
    )


# --- tokenizing ----------------------------------------------------------


def test_tokenizing_lowercases_and_splits_on_punctuation() -> None:
    assert tokenize("Traffic Engineering, with RL!") == ["traffic", "engineering", "with", "rl"]


def test_an_acronym_splits_the_same_way_in_text_and_query() -> None:
    """`CFR-RL` has to match however the paper happens to punctuate it."""
    assert tokenize("CFR-RL") == ["cfr", "rl"]
    assert tokenize("CFR RL") == tokenize("CFR-RL") == tokenize("cfr/rl")


def test_accents_are_folded() -> None:
    """Half the §12.2 questions are Spanish; diacritics must not split a token."""
    assert tokenize("utilización") == tokenize("utilizacion") == ["utilizacion"]
    assert tokenize("¿Qué metodologías?") == ["que", "metodologias"]


def test_digits_survive_tokenizing() -> None:
    assert tokenize("Booksim 2.0 and 13.73%") == ["booksim", "2", "0", "and", "13", "73"]


def test_tokenizing_empty_or_symbolic_text_yields_nothing() -> None:
    assert tokenize("") == []
    assert tokenize("—— ¿? ...") == []


# --- the keyword index ---------------------------------------------------


CORPUS = [
    chunk("A:1:0", "CFR-RL reroutes only the critical flows selected by the RL agent."),
    chunk("B:1:0", "The Jellyfish topology is a random regular graph, so ECMP unbalances load."),
    chunk("C:1:0", "Ee-Routing classifies traffic into elephant flows and mice flows."),
    chunk("D:1:0", "A survey of data center network architectures and their properties."),
]


def test_a_keyword_search_finds_the_chunk_holding_the_term() -> None:
    index = KeywordIndex(CORPUS)
    assert [c.id for c in index.search("Ee-Routing elephant flows", limit=1)] == ["C:1:0"]


def test_a_keyword_search_finds_an_acronym_a_vector_might_miss() -> None:
    index = KeywordIndex(CORPUS)
    assert index.search("ECMP", limit=3)[0].id == "B:1:0"


def test_chunks_matching_nothing_are_not_padded_in() -> None:
    """Ranking a chunk that holds no query term would be a lie to the fusion step."""
    index = KeywordIndex(CORPUS)
    hits = index.search("Jellyfish", limit=4)
    assert [c.id for c in hits] == ["B:1:0"]


def test_a_query_with_no_corpus_term_returns_nothing() -> None:
    index = KeywordIndex(CORPUS)
    assert index.search("quantum entanglement", limit=5) == []


def test_a_query_of_pure_punctuation_returns_nothing() -> None:
    assert KeywordIndex(CORPUS).search("¿...?", limit=5) == []


def test_an_empty_corpus_does_not_raise() -> None:
    """BM25 divides by the average document length, so this is a real edge case."""
    index = KeywordIndex([])
    assert len(index) == 0
    assert index.search("anything", limit=5) == []


def test_blank_chunks_are_excluded_from_the_statistics() -> None:
    index = KeywordIndex([*CORPUS, chunk("E:1:0", "   ")])
    assert len(index) == len(CORPUS)


def test_a_limit_of_zero_short_circuits() -> None:
    assert KeywordIndex(CORPUS).search("CFR-RL", limit=0) == []


def test_the_keyword_index_reads_the_whole_store(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    store.add(
        ids=["A:1:0", "B:1:0"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["fog computing at the edge", "reinforcement learning for routing"],
        metadatas=[{"attachment_key": "A", "page": 1}, {"attachment_key": "B", "page": 1}],
    )

    index = KeywordIndex.from_store(store)

    assert len(index) == 2
    assert index.search("fog", limit=1)[0].id == "A:1:0"


# --- fusion --------------------------------------------------------------


def test_fusion_rewards_a_chunk_both_retrievers_found() -> None:
    """The whole point: agreement between two different signals outranks either."""
    vector = [chunk("A:1:0"), chunk("B:1:0"), chunk("C:1:0")]
    keyword = [chunk("C:1:0"), chunk("D:1:0"), chunk("A:1:0")]

    fused = reciprocal_rank_fusion([vector, keyword], limit=4)

    # A is 1st + 3rd, C is 3rd + 1st — both beat B (2nd only) and D (2nd only).
    assert set(c.id for c in fused[:2]) == {"A:1:0", "C:1:0"}


def test_fusion_scores_are_the_sum_of_reciprocal_ranks() -> None:
    vector = [chunk("A:1:0"), chunk("B:1:0")]
    keyword = [chunk("B:1:0")]
    fused = reciprocal_rank_fusion([vector, keyword], limit=2)

    # B: 1/(k+2) + 1/(k+1) beats A: 1/(k+1) alone.
    assert 1 / (RRF_K + 2) + 1 / (RRF_K + 1) > 1 / (RRF_K + 1)
    assert [c.id for c in fused] == ["B:1:0", "A:1:0"]


def test_fusion_respects_the_limit() -> None:
    ranking = [chunk(f"A:{n}:0") for n in range(10)]
    assert len(reciprocal_rank_fusion([ranking], limit=3)) == 3


def test_fusion_keeps_the_copy_carrying_a_distance() -> None:
    """BM25 returns the same chunk without a distance; the citation needs the hit's."""
    vector = [chunk("A:1:0", distance=0.25)]
    keyword = [chunk("A:1:0", distance=None)]

    fused = reciprocal_rank_fusion([keyword, vector], limit=1)

    assert len(fused) == 1
    assert fused[0].distance == 0.25


def test_fusion_is_deterministic_when_scores_tie() -> None:
    first = reciprocal_rank_fusion([[chunk("B:1:0"), chunk("A:1:0")]], limit=2)
    second = reciprocal_rank_fusion([[chunk("B:1:0"), chunk("A:1:0")]], limit=2)
    assert [c.id for c in first] == [c.id for c in second]


def test_fusion_of_nothing_is_nothing() -> None:
    assert reciprocal_rank_fusion([], limit=5) == []
    assert reciprocal_rank_fusion([[], []], limit=5) == []


# --- candidate depth -----------------------------------------------------


def test_candidates_run_deeper_than_the_final_cut() -> None:
    """Fusing two `top_k`-long lists would give the merge nothing to work with."""
    assert candidate_depth(10) == 10 * CANDIDATE_MULTIPLIER
    assert candidate_depth(1) == MIN_CANDIDATES
    assert candidate_depth(5) >= MIN_CANDIDATES


# --- retrieve() ----------------------------------------------------------


@pytest.fixture
def hybrid_store(tmp_path: Path) -> VectorStore:
    """Three chunks whose embeddings and wording disagree about relevance."""
    store = VectorStore(tmp_path / "chroma")
    store.add(
        ids=["A:1:0", "B:1:0", "C:1:0"],
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]],
        documents=[
            "A general discussion of load balancing in networks.",
            "Another general discussion of balancing traffic.",
            "ECMP is the equal-cost multi-path default in this scheme.",
        ],
        metadatas=[{"attachment_key": key, "page": 1, "chunk_index": 0} for key in ("A", "B", "C")],
    )
    return store


def test_vector_mode_ignores_the_keyword_index(hybrid_store: VectorStore) -> None:
    index = KeywordIndex.from_store(hybrid_store)
    hits = retrieve(
        hybrid_store, [1.0, 0.0, 0.0], "ECMP", top_k=2, mode="vector", keyword_index=index
    )
    assert [c.id for c in hits] == ["A:1:0", "B:1:0"]


def test_hybrid_surfaces_a_chunk_the_vectors_ranked_last(hybrid_store: VectorStore) -> None:
    """The case §11.4 predicted: the term is in C, but C is the furthest vector."""
    index = KeywordIndex.from_store(hybrid_store)

    hits = retrieve(
        hybrid_store, [1.0, 0.0, 0.0], "ECMP", top_k=2, mode="hybrid", keyword_index=index
    )

    assert "C:1:0" in [c.id for c in hits]


def test_hybrid_without_an_index_falls_back_to_vectors(hybrid_store: VectorStore) -> None:
    hits = retrieve(hybrid_store, [1.0, 0.0, 0.0], "ECMP", top_k=1, mode="hybrid")
    assert [c.id for c in hits] == ["A:1:0"]


def test_hybrid_falls_back_when_bm25_matches_nothing(hybrid_store: VectorStore) -> None:
    """A question sharing no token with the corpus must not return an empty list."""
    index = KeywordIndex.from_store(hybrid_store)

    hits = retrieve(
        hybrid_store,
        [1.0, 0.0, 0.0],
        "quantum entanglement",
        top_k=2,
        mode="hybrid",
        keyword_index=index,
    )

    assert [c.id for c in hits] == ["A:1:0", "B:1:0"]


def test_hybrid_returns_at_most_top_k(hybrid_store: VectorStore) -> None:
    index = KeywordIndex.from_store(hybrid_store)
    hits = retrieve(
        hybrid_store, [1.0, 0.0, 0.0], "balancing ECMP", top_k=2, mode="hybrid", keyword_index=index
    )
    assert len(hits) == 2


# --- through the query pipeline ------------------------------------------


def test_hybrid_composes_with_windowing(tmp_path: Path) -> None:
    """Fusion picks the hits; windowing then expands whatever survived."""
    store = VectorStore(tmp_path / "chroma")
    store.add(
        ids=["A:1:0", "A:1:1", "A:1:2"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        documents=["before the term", "the term is rEDKSP here", "after the term"],
        metadatas=[{"attachment_key": "A", "page": 1, "chunk_index": index} for index in range(3)],
    )
    index = KeywordIndex.from_store(store)
    seen: list[str] = []

    def answer(messages: list) -> object:  # type: ignore[type-arg]
        from langchain_core.messages import AIMessage

        seen.append(str(messages[-1].content))
        return AIMessage(content="Because [1].")

    result = answer_question(
        "what is rEDKSP?",
        store=store,
        # A query vector pointing at the *first* chunk, so only BM25 finds the term.
        embed_query=lambda _: [1.0, 0.0, 0.0],
        answer=answer,  # type: ignore[arg-type]
        top_k=1,
        window=1,
        mode="hybrid",
        keyword_index=index,
    )

    assert result.chunks_retrieved == 1
    assert result.chunks_in_context == 3
    assert "rEDKSP" in seen[0]


def test_a_match_scoring_zero_is_still_a_match() -> None:
    """rank_bm25's IDF is exactly 0 when a term is in half the corpus.

    Filtering on `score > 0` would silently drop these, which on a small store is
    most of them.
    """
    corpus = [chunk("A:1:0", "fog computing"), chunk("B:1:0", "cloud routing")]
    index = KeywordIndex(corpus)

    hits = index.search("fog", limit=2)

    assert [c.id for c in hits] == ["A:1:0"]
    # Proof the score really is zero, so the test is exercising what it claims.
    from rank_bm25 import BM25Okapi

    assert BM25Okapi([tokenize(c.text) for c in corpus]).get_scores(["fog"])[0] == 0.0
