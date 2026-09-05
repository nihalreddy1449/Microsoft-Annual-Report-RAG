"""Hybrid retrieval: BM25 and vector search fused by Reciprocal Rank Fusion.

The two retrievers fail in opposite directions, which is the whole reason to
run both. Measured on this corpus, asked for FY2025 revenue, dense retrieval
put the table holding 281,724 at rank 6 behind prose reading "Revenue increased
$36.6 billion or 15%" - good matches on meaning, but a bare numeral carries
almost nothing for an embedder to grip. BM25 has the opposite problem: it finds
281,724 instantly and is helpless on "how much money did Microsoft make", which
shares no words with the text that answers it.

**Why RRF rather than blending scores.** BM25 scores are unbounded and
corpus-dependent; cosine similarities live in [-1, 1] and, on this corpus, sit
in a narrow band around 0.6-0.8. Adding or averaging them requires inventing a
normalisation, and whatever you choose silently becomes a tuning knob that
decides the outcome. RRF ignores the scores entirely and uses only rank
position:

    score(chunk) = sum over retrievers of  1 / (k + rank)

A chunk ranked highly by either retriever scores well; one ranked highly by
both wins. There is nothing to normalise and nothing to tune per corpus, which
is what makes the ablation's retrieval stage comparable across strategies.

k=60 is the value from the original RRF paper (Cormack et al., 2009) and is
the conventional default. It damps the difference between the top ranks, so
rank 1 does not overwhelm rank 2-3 from the other retriever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

RRF_K = 60


@dataclass
class FusedHit:
    chunk_id: str
    text: str
    score: float  # RRF score; comparable within one query only
    metadata: dict[str, Any]
    vector_rank: int | None = None
    bm25_rank: int | None = None

    @property
    def found_by(self) -> str:
        if self.vector_rank is not None and self.bm25_rank is not None:
            return "both"
        return "vector" if self.vector_rank is not None else "bm25"


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """Fuse named rankings of chunk ids into one score per id.

    ``rankings`` maps a retriever name to its ordered list of chunk ids.
    """
    scores: dict[str, float] = {}
    for ids in rankings.values():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Vector + BM25 retrieval over one chunking strategy, fused with RRF."""

    def __init__(self, vector_store, bm25_index, strategy: str) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self.strategy = strategy

    def search(
        self,
        question: str,
        k: int = 10,
        fiscal_years: Iterable[int] | None = None,
        candidates: int | None = None,
    ) -> list[FusedHit]:
        """Retrieve k chunks, fusing both retrievers.

        Each retriever contributes ``candidates`` results before fusion. Pulling
        deeper than k matters: a chunk that BM25 ranks 15th and the vector store
        ranks 12th may deserve to be in the final top 5, but only if both lists
        reach far enough down to see it.
        """
        depth = candidates or max(k * 4, 20)

        vector_hits = self.vector_store.query(
            self.strategy, question, k=depth, fiscal_years=fiscal_years
        )
        bm25_hits = self.bm25.search(question, k=depth, fiscal_years=fiscal_years)

        rankings = {
            "vector": [h.chunk_id for h in vector_hits],
            "bm25": [h.chunk_id for h in bm25_hits],
        }
        fused = reciprocal_rank_fusion(rankings)

        vector_rank = {h.chunk_id: i + 1 for i, h in enumerate(vector_hits)}
        bm25_rank = {h.chunk_id: i + 1 for i, h in enumerate(bm25_hits)}
        lookup = {h.chunk_id: (h.text, h.metadata) for h in vector_hits}
        lookup.update({h.chunk_id: (h.text, h.metadata) for h in bm25_hits})

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        results: list[FusedHit] = []
        for chunk_id, score in ordered:
            text, metadata = lookup[chunk_id]
            results.append(
                FusedHit(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    metadata=metadata,
                    vector_rank=vector_rank.get(chunk_id),
                    bm25_rank=bm25_rank.get(chunk_id),
                )
            )
        return results
