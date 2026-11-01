"""Retrieval strategies beyond plain vector similarity (M7, §11.4).

Embeddings are good at meaning and bad at strings. A question naming `CFR-RL`,
`ECMP`, `rEDKSP` or `Ee-Routing` is asking about a token, not a concept, and a
dense vector will happily return a passage that is *about the same subject* while
missing the one that contains the term. BM25 has the opposite failure: it cannot
match a paraphrase at all. Fusing the two ranked lists keeps each one's strength.

Nothing here touches the store's contents. BM25 is computed at query time from the
text already in Chroma, so switching `retrieval` needs no re-indexing and cannot
invalidate a fingerprint — unlike a changed `embedding_model` (§11.1).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from typing import Literal

from rank_bm25 import BM25Okapi

from zotero_rag_core.store import ChunkStore, Scope, StoredChunk, in_scope

logger = logging.getLogger(__name__)

RetrievalMode = Literal["vector", "hybrid"]

#: Candidates taken from *each* retriever before fusion. Fusing two `top_k`-long
#: lists would give the merge almost nothing to work with — the interesting case is
#: a chunk ranked 14th by vectors and 2nd by BM25, which only appears if both lists
#: run deeper than the final cut.
CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATES = 20

#: The conventional reciprocal-rank-fusion constant. It damps the difference
#: between adjacent ranks, so a list that is confidently wrong at rank 1 cannot
#: dominate a list that is merely right at rank 3.
RRF_K = 60

_TOKEN = re.compile(r"[a-z0-9]+")


def candidate_depth(top_k: int) -> int:
    """How deep each retriever runs before the lists are fused."""
    return max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)


def tokenize(text: str) -> list[str]:
    """Fold accents, lowercase, and split on anything not alphanumeric.

    Accent folding is what lets a Spanish question match Spanish text that was
    typed, OCR'd or copied with different diacritics — `utilización` and
    `utilizacion` have to be one token or half the §12.2 questions retrieve nothing.

    Splitting on punctuation means `CFR-RL` becomes `["cfr", "rl"]`. That is
    deliberate: the query is split the same way, so the pair still matches, and it
    also catches the paper that writes the name as `CFR RL` or `CFR/RL`.

    No stopword list. BM25's IDF already discounts words that appear everywhere,
    and a fixed list would have to pick a language — wrong for a corpus queried in
    Spanish and written in English.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _TOKEN.findall(unaccented.casefold())


class KeywordIndex:
    """BM25 over the text of every stored chunk.

    Built in memory from the store rather than persisted: the whole corpus is
    already on disk in Chroma, and a second copy would be one more thing to keep in
    step with the index. Construction cost is a tokenize pass over every chunk, so
    build it **once per process** and reuse it — `eval` over 48 questions must not
    rebuild it 48 times.
    """

    def __init__(self, chunks: Sequence[StoredChunk]) -> None:
        # An empty document would contribute nothing and BM25Okapi divides by the
        # average length, so drop blanks rather than skew the statistics.
        self._chunks = [chunk for chunk in chunks if chunk.text.strip()]
        corpus = [tokenize(chunk.text) for chunk in self._chunks]
        # Kept so "did this chunk match?" can be answered exactly; see `search`.
        self._tokens = [set(document) for document in corpus]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        logger.info("keyword index built over %d chunk(s)", len(self._chunks))

    @classmethod
    def from_store(cls, store: ChunkStore) -> KeywordIndex:
        return cls(store.all_chunks())

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, question: str, limit: int, scope: Scope = None) -> list[StoredChunk]:
        """The `limit` best-matching chunks within `scope`, best first.

        The scope is applied before ranking, so a narrow folder still gets its full
        `limit` rather than whatever survived a whole-store cut. The BM25 statistics
        stay those of the whole store: the index is built once per session, and IDF
        over the whole library is, if anything, the better estimate of how rare a
        term is.

        Chunks containing none of the query's tokens are dropped rather than padded
        in: ranking them would hand fusion a confident-looking position for a
        passage BM25 never matched.

        The test for "matched" is token presence, **not** a positive score. It is
        tempting to filter on `score > 0`, and it is wrong: `rank_bm25` uses the
        Robertson–Sparck-Jones IDF, `log(N - df + 0.5) - log(df + 0.5)`, which is
        exactly **zero** when a term appears in half the documents and negative above
        that. So a genuine match can score zero — reliably so on a small store, where
        `df = N/2` is easy to hit — and filtering on the score silently returns
        nothing. A four-chunk collection found this before a user did.
        """
        if self._bm25 is None or limit <= 0 or (scope is not None and not scope):
            return []
        tokens = tokenize(question)
        if not tokens:
            return []

        wanted = set(tokens)
        matched = [
            index
            for index, present in enumerate(self._tokens)
            if not wanted.isdisjoint(present) and in_scope(self._chunks[index].metadata, scope)
        ]
        if not matched:
            return []

        scores = self._bm25.get_scores(tokens)
        # Ties broken by id, so two runs of one question rank identically.
        matched.sort(key=lambda index: (-scores[index], self._chunks[index].id))
        return [self._chunks[index] for index in matched[:limit]]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[StoredChunk]], limit: int
) -> list[StoredChunk]:
    """Merge ranked lists by summing `1 / (RRF_K + rank)` over the lists.

    Rank-based rather than score-based on purpose: a cosine distance and a BM25
    score have no common scale, and any attempt to normalise them — min-max over the
    candidates, say — makes the fused order depend on how extreme the worst
    candidate happened to be. Positions are comparable; magnitudes are not.
    """
    scores: dict[str, float] = {}
    best: dict[str, StoredChunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)
            # Keep whichever copy carries a similarity distance: the same chunk
            # arrives from BM25 without one, and `Source.distance` should survive.
            current = best.get(chunk.id)
            if current is None or (current.distance is None and chunk.distance is not None):
                best[chunk.id] = chunk

    order = sorted(best, key=lambda identifier: (-scores[identifier], identifier))
    return [best[identifier] for identifier in order[:limit]]


def retrieve(
    store: ChunkStore,
    embedding: list[float],
    question: str,
    top_k: int,
    mode: RetrievalMode = "vector",
    keyword_index: KeywordIndex | None = None,
    scope: Scope = None,
) -> list[StoredChunk]:
    """The `top_k` chunks for a question within `scope`, by the configured strategy.

    Both retrievers apply the scope themselves, before their own cut, so fusion only
    ever sees in-scope candidates.

    Falls back to plain vector retrieval when hybrid is asked for but no keyword
    index was supplied, or when BM25 matched nothing at all: a question sharing no
    token with the corpus should return the vector result, not an empty list.
    """
    if mode != "hybrid" or keyword_index is None:
        return store.query(embedding, top_k, scope)

    depth = candidate_depth(top_k)
    vector_hits = store.query(embedding, depth, scope)
    keyword_hits = keyword_index.search(question, depth, scope)

    if not keyword_hits:
        logger.info("hybrid: BM25 matched nothing, using vector results alone")
        return vector_hits[:top_k]

    fused = reciprocal_rank_fusion([vector_hits, keyword_hits], top_k)
    logger.info(
        "hybrid: %d vector + %d keyword candidate(s) fused to %d",
        len(vector_hits),
        len(keyword_hits),
        len(fused),
    )
    return fused
