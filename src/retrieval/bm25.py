"""BM25 sparse retrieval over the same chunks the vector store holds.

BM25 scores a document by how often the query's terms appear in it, weighted so
that rare terms count for more and long documents are not unfairly rewarded. It
matches *strings*, not meaning, which makes it the exact complement to dense
retrieval: the vector store handles "how much money did Microsoft make", BM25
handles "281,724".

Tokenisation is the part that matters on this corpus, and a default tokeniser
gets it wrong in ways that are easy to miss. Splitting on non-alphanumerics
turns "281,724" into "281" and "724" - two meaningless tokens that collide with
unrelated numbers all over the filings - and "13.64" into "13" and "64". The
one advantage BM25 has over embeddings is exact matching of figures, and naive
tokenisation throws it away.

So numbers are kept whole, including their separators and trailing percent
sign, and a comma-stripped variant is indexed alongside so a query written
"281724" still matches a document written "281,724".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

# A number (with , . % attached) or a word. Ordered so numbers win.
_TOKEN = re.compile(r"\d[\d,.]*%?|[a-z][a-z']*")

# Words so common in these filings that they carry no retrieval signal.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with we our us""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, keep figures whole, and index a comma-free numeric variant."""
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        token = token.rstrip(".")
        if not token or token in _STOPWORDS:
            continue
        tokens.append(token)
        if "," in token:
            # "281,724" is also findable as "281724".
            tokens.append(token.replace(",", ""))
    return tokens


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class BM25Index:
    """In-memory BM25 over one chunking strategy's chunks.

    Built on load rather than persisted: 2,000-2,400 chunks index in well under
    a second, and a stale index that silently disagrees with the vector store
    would be a far worse problem than the rebuild cost.
    """

    def __init__(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.ids = [c["chunk_id"] for c in chunks]
        self._corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(
        self,
        question: str,
        k: int = 10,
        fiscal_years: Iterable[int] | None = None,
    ) -> list[Hit]:
        """Top-k chunks by BM25 score, optionally restricted to fiscal years."""
        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        allowed = {int(y) for y in fiscal_years} if fiscal_years else None
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        hits: list[Hit] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            chunk = self.chunks[i]
            if allowed is not None and int(chunk["fiscal_year"]) not in allowed:
                continue
            hits.append(
                Hit(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=float(scores[i]),
                    metadata=chunk,
                )
            )
            if len(hits) >= k:
                break
        return hits
