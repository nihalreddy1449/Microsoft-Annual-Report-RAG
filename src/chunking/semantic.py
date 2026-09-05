"""Semantic chunking - place boundaries where the topic actually changes.

Each atomic unit is embedded, and the cosine similarity between consecutive
units is read as a signal of topical continuity. Where similarity dips, the
subject has moved on and a boundary is placed there.

The threshold is a percentile of each document's own similarity distribution
rather than a fixed number. Absolute similarity varies a lot by content -
runs of dense financial prose sit high, a shift from narrative into a table
sits low - so a constant threshold that suits the MD&A would carve the notes
to pieces. A percentile adapts per document and keeps the strategy comparable
across all six.

Deliberately *not* special-cased: tables are given no forced boundary. A table
embeds very differently from the prose around it, so the similarity signal puts
a boundary there on its own. Hard-coding one would quietly import structural
knowledge into the arm that is supposed to be the structure-free comparison,
and would blur what separates this strategy from structure-aware chunking.

Atomicity and the size budget still apply, as they do in every arm: units are
never split, and an over-budget segment is divided at its own weakest internal
similarity rather than at an arbitrary point.
"""

from __future__ import annotations

import numpy as np

from ..embedding.embedder import Embedder, cosine_similarity_pairs
from .base import AtomicUnit, Chunk, assemble

DEFAULT_MAX_CHARS = 1800
DEFAULT_PERCENTILE = 25.0  # split at the lowest quarter of similarities

_shared_embedder: Embedder | None = None


def _embedder() -> Embedder:
    """One shared model instance; loading it per document would dominate runtime."""
    global _shared_embedder
    if _shared_embedder is None:
        _shared_embedder = Embedder()
    return _shared_embedder


def chunk_semantic(
    units: list[AtomicUnit],
    max_chars: int = DEFAULT_MAX_CHARS,
    percentile: float = DEFAULT_PERCENTILE,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Group units into chunks at topical boundaries found by embedding similarity."""
    if not units:
        return []
    if len(units) == 1:
        chunk = assemble(units, "semantic", 0)
        return [chunk] if chunk else []

    model = embedder or _embedder()
    vectors = model.encode_documents([u.text for u in units])
    similarities = cosine_similarity_pairs(vectors)

    # A boundary sits after unit i when similarity(i, i+1) is in the lowest
    # `percentile` of this document's similarities.
    threshold = float(np.percentile(similarities, percentile))
    boundaries = {i for i, s in enumerate(similarities) if s <= threshold}

    segments = _merge_to_budget(_segment(units, boundaries), max_chars)

    chunks: list[Chunk] = []
    for segment in segments:
        for piece in _enforce_budget(segment, similarities, units, max_chars):
            chunk = assemble(piece, "semantic", len(chunks))
            if chunk is not None:
                chunks.append(chunk)
    return chunks


def _merge_to_budget(
    segments: list[list[AtomicUnit]], max_chars: int
) -> list[list[AtomicUnit]]:
    """Recombine consecutive segments while they fit the budget.

    Splitting at every dip below the threshold leaves mostly very short
    segments: a quarter of all gaps become boundaries, so the mean chunk was
    ~770 characters and barely responded to the threshold at all, because size
    was really being set by budget-splitting rather than by topic.

    Merging back up to the budget restores the intent. Boundaries still land
    only where similarity dipped - the merge never invents one - but adjacent
    fragments of the same topic are rejoined instead of being embedded as
    isolated sentences. It also brings mean chunk size into the same band as
    the other two arms, so recall@k compares boundary placement rather than
    how much text each arm happens to return.
    """
    merged: list[list[AtomicUnit]] = []
    for segment in segments:
        size = sum(u.char_count for u in segment)
        if merged and sum(u.char_count for u in merged[-1]) + size <= max_chars:
            merged[-1].extend(segment)
        else:
            merged.append(list(segment))
    return merged


def _segment(units: list[AtomicUnit], boundaries: set[int]) -> list[list[AtomicUnit]]:
    segments: list[list[AtomicUnit]] = []
    current: list[AtomicUnit] = []
    for i, unit in enumerate(units):
        current.append(unit)
        if i in boundaries:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _enforce_budget(
    segment: list[AtomicUnit],
    similarities: np.ndarray,
    all_units: list[AtomicUnit],
    max_chars: int,
) -> list[list[AtomicUnit]]:
    """Split an over-budget segment at its weakest internal seam, recursively.

    Cutting at the lowest remaining similarity keeps the split faithful to the
    strategy: even when size forces a break, it lands where the text is least
    connected rather than wherever the budget ran out.
    """
    size = sum(u.char_count for u in segment)
    if size <= max_chars or len(segment) == 1:
        return [segment]

    position = {u.index: i for i, u in enumerate(all_units)}
    start = position[segment[0].index]

    # Candidate seams inside this segment, weakest first.
    seams = [
        (float(similarities[start + offset]), offset)
        for offset in range(len(segment) - 1)
        if start + offset < len(similarities)
    ]
    if not seams:
        return [segment]

    _, cut = min(seams)
    left, right = segment[: cut + 1], segment[cut + 1 :]
    if not left or not right:
        return [segment]

    return _enforce_budget(left, similarities, all_units, max_chars) + _enforce_budget(
        right, similarities, all_units, max_chars
    )


__all__ = ["chunk_semantic", "DEFAULT_MAX_CHARS", "DEFAULT_PERCENTILE"]
