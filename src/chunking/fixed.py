"""Fixed-size chunking - the ablation baseline.

This is deliberately the naive strategy: walk the document as one character
stream and cut every N characters with a fixed overlap, paying no attention to
headings, paragraphs or sentences. Cuts land mid-sentence, and a chunk happily
spans the boundary between two unrelated sections. That is the point - it is
the control the other two arms are measured against.

The one concession is decision 4: tables and bullet groups are atomic in every
arm, so a cut that would land inside one is pushed to that unit's end. Without
that the comparison would be confounded, because the baseline would be
degrading table rows while the other arms kept them intact, and the measured
difference would no longer isolate the chunking strategy.

Paragraphs get no such protection, which is the honest reading of "fixed-size".
"""

from __future__ import annotations

from .base import AtomicUnit, Chunk, assemble, context_header

# Calibrated so all three strategies emit chunks of nearly the same mean size
# (fixed 970, structure-aware 986, semantic 999 characters). The knobs differ
# a lot between arms - 800 here against 1400 and 1800 elsewhere - because each
# strategy packs text differently: this one overruns its budget whenever a cut
# would land inside an atomic unit, while the others fall short of theirs
# whenever a section or topic ends early. Matching the knobs would therefore
# have produced badly unmatched chunks. It is the output that has to match,
# otherwise recall@k would partly measure how much text each arm returns
# rather than where it places boundaries. See scripts/run_chunking.py.
DEFAULT_CHUNK_CHARS = 800
DEFAULT_OVERLAP_CHARS = 150

_SEPARATOR = "\n\n"


def chunk_fixed(
    units: list[AtomicUnit],
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Cut the document into fixed-length windows with overlap."""
    if not units:
        return []
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    # Flatten to one stream, remembering where each unit sits so a cut can be
    # tested against the units it would fall inside.
    pieces: list[str] = []
    spans: list[tuple[int, int, AtomicUnit]] = []
    cursor = 0
    for unit in units:
        if pieces:
            cursor += len(_SEPARATOR)
            pieces.append(_SEPARATOR)
        start = cursor
        pieces.append(unit.text)
        cursor += len(unit.text)
        spans.append((start, cursor, unit))
    stream = "".join(pieces)

    def unit_at(position: int) -> tuple[int, int, AtomicUnit] | None:
        for span in spans:
            if span[0] <= position < span[1]:
                return span
        return None

    def units_between(start: int, end: int) -> list[AtomicUnit]:
        return [u for (s, e, u) in spans if s < end and e > start]

    chunks: list[Chunk] = []
    position = 0
    while position < len(stream):
        cut = min(position + chunk_chars, len(stream))

        # Never cut inside an atomic unit; extend to its end instead.
        if cut < len(stream):
            landed = unit_at(cut)
            if landed is not None and landed[2].kind in ("table", "list"):
                cut = landed[1]

        covered = units_between(position, cut)
        if not covered:  # pathological, but keeps the loop total
            covered = [spans[0][2]]

        first = covered[0]
        header = context_header(first.fiscal_year, first.section, first.subsection)
        body = stream[position:cut].strip()
        if body:
            chunks.append(
                Chunk(
                    chunk_id=f"fixed-FY{first.fiscal_year}-{len(chunks):04d}",
                    text=f"{header}\n\n{body}",
                    strategy="fixed",
                    fiscal_year=first.fiscal_year,
                    source=first.source,
                    # A fixed window can straddle sections; it is labelled with
                    # where it starts, which is part of what the baseline costs.
                    section=first.section,
                    subsection=first.subsection,
                    kind=_kind_of(covered),
                    unit_indices=[u.index for u in covered],
                )
            )

        if cut >= len(stream):
            break
        position = max(cut - overlap_chars, position + 1)

    return chunks


def _kind_of(units: list[AtomicUnit]) -> str:
    kinds = {u.kind for u in units}
    if kinds == {"table"}:
        return "table"
    return "mixed" if "table" in kinds else "prose"


__all__ = ["chunk_fixed", "DEFAULT_CHUNK_CHARS", "DEFAULT_OVERLAP_CHARS", "assemble"]
