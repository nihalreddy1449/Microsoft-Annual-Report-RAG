"""Chroma vector store - one collection per chunking strategy.

Chroma runs embedded, in-process, like SQLite: no server, no Docker, no network.
`PersistentClient` writes a directory of files under chroma_db/ and that is the
whole database. It is gitignored because it is derived data, fully rebuildable
from data/raw/ by re-running the pipeline.

Two decisions here matter more than they look:

**We pass our own vectors rather than letting Chroma embed.** Chroma will
happily embed text for you using its default model (all-MiniLM-L6-v2), which
it downloads on first use. Letting it do that would silently split the project
across two different embedding models - bge-base for the chunks we embedded
ourselves, MiniLM for anything Chroma touched - and vectors from different
models are not comparable. The failure mode is not a crash; it is quietly
meaningless similarity scores. So every call passes explicit embeddings.

**Queries and documents are embedded differently, and the store enforces it.**
bge wants a retrieval instruction prefixed to queries but not to passages.
`add_chunks` uses encode_documents, `query` uses encode_queries. Getting this
backwards costs recall with nothing to notice.

The collection is created with cosine space to match the L2-normalised vectors
the embedder produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..embedding.embedder import Embedder

DEFAULT_DB_PATH = "chroma_db"
DEFAULT_BATCH = 256


@dataclass
class Hit:
    """One retrieved chunk, with its similarity score."""

    chunk_id: str
    text: str
    score: float  # cosine similarity in [-1, 1]; higher is better
    metadata: dict[str, Any]

    @property
    def fiscal_year(self) -> int:
        return int(self.metadata["fiscal_year"])

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", ""))


def _clean_metadata(chunk: dict) -> dict[str, Any]:
    """Chroma accepts only str/int/float/bool, so flatten and drop None."""
    meta = {
        "strategy": chunk["strategy"],
        "fiscal_year": int(chunk["fiscal_year"]),
        "source": chunk["source"],
        "section": chunk["section"] or "",
        "subsection": chunk["subsection"] or "",
        "kind": chunk["kind"],
        "char_count": int(chunk.get("char_count", len(chunk["text"]))),
    }
    return {k: v for k, v in meta.items() if v is not None}


class VectorStore:
    """Persistent Chroma collections, one per chunking strategy."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, embedder: Embedder | None = None):
        self.db_path = Path(db_path)
        self.embedder = embedder or Embedder()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import chromadb

            self.db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.db_path))
        return self._client

    def collection(self, strategy: str):
        """Get or create the collection for one chunking strategy."""
        return self.client.get_or_create_collection(
            name=f"chunks_{strategy}",
            # Cosine, to match the L2-normalised vectors bge produces. Chroma's
            # default is squared L2, which ranks differently.
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self, strategy: str) -> None:
        """Drop a collection so indexing is idempotent across re-runs."""
        try:
            self.client.delete_collection(f"chunks_{strategy}")
        except Exception:  # noqa: BLE001 - absent collection is fine
            pass

    def add_chunks(
        self,
        strategy: str,
        chunks: list[dict],
        batch_size: int = DEFAULT_BATCH,
        progress: bool = False,
    ) -> int:
        """Embed and store chunks. Returns the number written."""
        collection = self.collection(strategy)
        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.encode_documents([c["text"] for c in batch])
            collection.add(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["text"] for c in batch],
                embeddings=[v.tolist() for v in vectors],
                metadatas=[_clean_metadata(c) for c in batch],
            )
            written += len(batch)
            if progress:
                print(f"    {written}/{len(chunks)}", flush=True)
        return written

    def query(
        self,
        strategy: str,
        question: str,
        k: int = 10,
        fiscal_years: Iterable[int] | None = None,
    ) -> list[Hit]:
        """Retrieve the k nearest chunks, optionally restricted to fiscal years."""
        where: dict[str, Any] | None = None
        if fiscal_years:
            years = [int(y) for y in fiscal_years]
            where = {"fiscal_year": {"$in": years}} if len(years) > 1 else {"fiscal_year": years[0]}

        vector = self.embedder.encode_queries([question])[0]
        result = self.collection(strategy).query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[Hit] = []
        for chunk_id, doc, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            # Chroma returns cosine *distance*; similarity is 1 - distance.
            hits.append(Hit(chunk_id=chunk_id, text=doc, score=1.0 - float(distance), metadata=meta))
        return hits

    def count(self, strategy: str) -> int:
        return self.collection(strategy).count()
