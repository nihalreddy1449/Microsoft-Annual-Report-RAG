"""bge-base-en-v1.5 embedding wrapper.

Two details about this model family matter enough to be handled here rather
than left to each caller:

**Queries and documents are encoded differently.** BAAI recommends prefixing a
*query* with a short retrieval instruction while leaving passages bare. The
prefix nudges the query vector toward the region where relevant passages sit,
and it is worth a few points of recall on short queries - which is exactly what
this project's eval questions are. Encoding a query as though it were a
document is a silent quality loss with no error to notice, so `encode_queries`
and `encode_documents` are separate methods and the raw `encode` is internal.

**Vectors must be L2-normalised.** bge is trained for cosine similarity, and
normalising lets a vector store compute cosine as a plain dot product. Chroma
is configured for cosine distance to match; mismatching the two is another
silent-degradation failure, not a crash.

The model is ~110M parameters and loads to GPU in well under a second, sharing
the 8GB card comfortably with the reranker later.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

# BAAI's recommended retrieval instruction for the English v1.5 models.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Lazily-loaded sentence embedder with query/document asymmetry."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
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
        """Load on first use so importing this module stays cheap."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity becomes a dot product
            show_progress_bar=show_progress,
        )
        return vectors.astype(np.float32)

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode passages. No instruction prefix, per the model card."""
        return self._encode(texts, show_progress)

    def encode_queries(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode queries, with the retrieval instruction prefix bge expects."""
        return self._encode([QUERY_INSTRUCTION + t for t in texts], show_progress)


def cosine_similarity_pairs(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between each consecutive pair of rows.

    Inputs are already L2-normalised, so this is a row-wise dot product.
    Returns an array of length ``len(vectors) - 1``.
    """
    if len(vectors) < 2:
        return np.zeros(0, dtype=np.float32)
    return np.sum(vectors[:-1] * vectors[1:], axis=1).astype(np.float32)
