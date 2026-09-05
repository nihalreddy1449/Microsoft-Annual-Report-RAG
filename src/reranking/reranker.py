"""Cross-encoder reranking with bge-reranker-base.

Retrieval and reranking use different model architectures, and the difference
is the entire reason a reranker helps.

The embedder is a **bi-encoder**: it turns the question into a vector and each
chunk into a vector, separately, and compares them. A chunk's vector is
computed without ever seeing the question, so it has to be a summary good
enough for every possible question at once. That is what makes dense retrieval
cheap - chunks are embedded once, offline - and also what makes it blunt.

A **cross-encoder** reads the question and one chunk *together* in a single
forward pass, so attention runs across both. It can notice that
"Operating income: 2022 = 83,383" directly answers "what was operating income
in fiscal year 2022", which two independently-written summaries cannot express.
The cost is that it scores one pair at a time: it cannot be precomputed, so it
only runs on a shortlist.

This is why the pipeline retrieves deep and reranks narrow. Measured on this
corpus, the chunk holding 83,383 was ranked 23rd by hybrid retrieval, so
reranking a top-10 shortlist would never have seen it. Retrieval must reach
past where the answer actually sits before reranking can fix the order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

DEFAULT_MODEL = "BAAI/bge-reranker-base"


@dataclass
class RerankedHit:
    """A chunk with both its retrieval and its reranked position."""

    chunk_id: str
    text: str
    score: float  # cross-encoder relevance logit; higher is better
    metadata: dict[str, Any]
    retrieval_rank: int | None = None


class Reranker:
    """Lazily-loaded cross-encoder that reorders a retrieval shortlist."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._model = None

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
        return self._model

    def rerank(
        self,
        question: str,
        hits: Sequence[Any],
        top_k: int | None = None,
    ) -> list[RerankedHit]:
        """Score each hit against the question and return them reordered.

        ``hits`` may be any objects carrying ``chunk_id``, ``text`` and
        ``metadata`` - both the vector store and the hybrid retriever qualify.
        """
        if not hits:
            return []

        pairs = [(question, h.text) for h in hits]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked = [
            RerankedHit(
                chunk_id=h.chunk_id,
                text=h.text,
                score=float(score),
                metadata=h.metadata,
                retrieval_rank=i + 1,
            )
            for i, (h, score) in enumerate(zip(hits, scores))
        ]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked[:top_k] if top_k else reranked


class RerankingRetriever:
    """Retrieve a deep shortlist, then rerank it down to the final k."""

    def __init__(self, retriever, reranker: Reranker, candidates: int = 40) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.candidates = candidates

    def search(self, question: str, k: int = 10, **kwargs) -> list[RerankedHit]:
        shortlist = self.retriever.search(question, k=self.candidates, **kwargs)
        return self.reranker.rerank(question, shortlist, top_k=k)
