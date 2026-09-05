"""End-to-end RAG pipeline: retrieve, rerank, generate.

One configurable object rather than a script, because the ablation needs to
build several differently-configured pipelines and run the same eval set
through each. A configuration is exactly the row of the results table it
produces: chunking strategy, whether BM25 is fused in, whether reranking runs.

The Gradio app and the eval harness both drive this same class, so what the
demo answers is what the numbers were measured on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .embedding.embedder import Embedder
from .generation.generator import Answer, Generator
from .reranking.reranker import Reranker, RerankingRetriever
from .retrieval.bm25 import BM25Index
from .retrieval.hybrid import HybridRetriever
from .retrieval.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PipelineConfig:
    """One row of the ablation table."""

    strategy: str = "semantic"
    use_bm25: bool = True
    use_reranker: bool = True
    top_k: int = 5
    rerank_candidates: int = 40

    @property
    def label(self) -> str:
        retriever = "hybrid" if self.use_bm25 else "vector"
        return f"{self.strategy} + {retriever}" + (" + rerank" if self.use_reranker else "")


class _VectorOnly:
    """Adapter giving the vector store the same search() shape as the others."""

    def __init__(self, store: VectorStore, strategy: str) -> None:
        self.store = store
        self.strategy = strategy

    def search(self, question: str, k: int = 10, **kwargs) -> list[Any]:
        return self.store.query(self.strategy, question, k=k, **kwargs)


class RAGPipeline:
    """Retrieval, optional reranking, and generation under one configuration."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        root: Path = ROOT,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.root = root
        # Models are shared across configurations when several pipelines are
        # built for an ablation sweep; reloading bge per row would dominate
        # the runtime and pointlessly re-fill the GPU.
        self.embedder = embedder or Embedder()
        self.store = VectorStore(root / "chroma_db", embedder=self.embedder)
        self._reranker = reranker
        self._generator = generator
        self._retriever = None

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator()
        return self._generator

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    @property
    def retriever(self):
        if self._retriever is None:
            cfg = self.config
            if cfg.use_bm25:
                path = self.root / "data" / "processed" / f"chunks_{cfg.strategy}.json"
                chunks = json.loads(path.read_text(encoding="utf-8"))
                base = HybridRetriever(self.store, BM25Index(chunks), cfg.strategy)
            else:
                base = _VectorOnly(self.store, cfg.strategy)

            self._retriever = (
                RerankingRetriever(base, self.reranker, candidates=cfg.rerank_candidates)
                if cfg.use_reranker
                else base
            )
        return self._retriever

    def retrieve(self, question: str, fiscal_years: Iterable[int] | None = None) -> list[Any]:
        kwargs = {"fiscal_years": fiscal_years} if fiscal_years else {}
        return self.retriever.search(question, k=self.config.top_k, **kwargs)

    def ask(self, question: str, fiscal_years: Iterable[int] | None = None) -> Answer:
        """Retrieve context and generate a grounded answer."""
        hits = self.retrieve(question, fiscal_years)
        return self.generator.answer(question, hits)
