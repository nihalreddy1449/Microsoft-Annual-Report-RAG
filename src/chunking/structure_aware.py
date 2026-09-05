"""Structure-aware chunking - group units by the document's own headings.

This is the arm that motivated the whole custom parser. It respects the
boundaries the filing actually has: a chunk holds units from one subsection,
never straddles a heading, and carries that heading as metadata a retriever can
filter on.

Two failure modes it has to avoid, both of which would quietly make it look
worse than the baseline rather than better:

  * Tiny chunks. Sections like "CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS"
    hold a single sentence. Embedding a fifteen-word chunk gives a vector
    dominated by the context header. Consecutive subsections of the *same*
    section are merged while they fit under the size budget.
  * Oversized chunks. NOTES TO FINANCIAL STATEMENTS runs to ~66,000 characters.
    bge-base-en-v1.5 truncates at 512 tokens, so an unsplit section would be
    silently decapitated - most of it never embedded at all. Overlong groups
    are split at unit boundaries, which keeps tables and bullet groups whole.
"""

from __future__ import annotations

from .base import AtomicUnit, Chunk, assemble

# Deliberately the same character budget the fixed-size arm uses. If the arms
# ran at different granularities, a score difference could be an artifact of
# chunk size rather than of where the boundaries were placed, and the ablation
# would no longer isolate the strategy. Mean chunk size still differs between
# arms - structure-aware cuts early when a section ends - so it is reported
# alongside the scores rather than hidden.
DEFAULT_MAX_CHARS = 1400
DEFAULT_MIN_CHARS = 400


def chunk_structure_aware(
    units: list[AtomicUnit],
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Chunk along heading boundaries, merging small groups and splitting big ones."""
    if not units:
        return []

    # 1. Partition into runs sharing a (section, subsection) heading path.
    groups: list[list[AtomicUnit]] = []
    for unit in units:
        key = (unit.section, unit.subsection)
        if groups and (groups[-1][0].section, groups[-1][0].subsection) == key:
            groups[-1].append(unit)
        else:
            groups.append([unit])

    # 2. Merge consecutive small groups, but only within the same section, so a
    #    merge never blurs the top-level boundary the strategy exists to honour.
    merged: list[list[AtomicUnit]] = []
    for group in groups:
        size = sum(u.char_count for u in group)
        if (
            merged
            and merged[-1][0].section == group[0].section
            and sum(u.char_count for u in merged[-1]) < min_chars
            and sum(u.char_count for u in merged[-1]) + size <= max_chars
        ):
            merged[-1].extend(group)
        else:
            merged.append(list(group))

    # 3. Split anything still over budget, at unit boundaries only.
    chunks: list[Chunk] = []
    for group in merged:
        for piece in _split_to_budget(group, max_chars):
            chunk = assemble(piece, "structure_aware", len(chunks))
            if chunk is not None:
                chunks.append(chunk)
    return chunks


def _split_to_budget(units: list[AtomicUnit], max_chars: int) -> list[list[AtomicUnit]]:
    """Break a group into pieces under the budget without splitting a unit.

    A single unit larger than the budget - a long financial table, say - is
    emitted alone rather than cut, honouring decision 4. It will be truncated
    by the embedder, which is a known and accepted cost of keeping tables
    intact, and is the same in every arm.
    """
    pieces: list[list[AtomicUnit]] = []
    current: list[AtomicUnit] = []
    size = 0
    for unit in units:
        if current and size + unit.char_count > max_chars:
            pieces.append(current)
            current, size = [], 0
        current.append(unit)
        size += unit.char_count
    if current:
        pieces.append(current)
    return pieces


__all__ = ["chunk_structure_aware", "DEFAULT_MAX_CHARS", "DEFAULT_MIN_CHARS"]
