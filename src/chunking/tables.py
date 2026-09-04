"""Turn extracted table rows into text a sentence embedder can represent.

Decision 3 of the locked methodology: tables are linearized row-wise into
self-describing text rather than embedded as HTML or markdown, because
bge-base-en-v1.5 is trained on natural language and markup tokens waste its
representation on syntax the question will never contain.

The hard part is column alignment. These tables come from HTML, so a row's
cells do not line up with the header's:

    header  (10 cells): '(In millions...)' '' '2025' '' '' '2024' '' '' 'PercentageChange' ''
    Revenue (13 cells): 'Revenue' '' '$' '281,724' '' '' '$' '245,122' '' '' '' '15%' ''

colspan artifacts leave empty padding cells and put '$' in a cell of its own,
so naive positional mapping pairs '2025' with '$' and can hand one year's
figure to another year. Compacting away empty and symbol-only cells fixes most
of it - measured across the corpus, 76.5% of 1,385 data rows then align - but
23.5% do not, and those include genuine two-row headers such as the FY2020
stockholders' equity table, where row 0 holds "Shares | Amount" repeated and
the years sit in row 1 beneath them.

So alignment is never assumed. When a row's width matches the chosen header,
values are attributed to their columns explicitly. When it does not, the row is
emitted with its values in source order and no column claims at all. A vague
row is recoverable by a reader; a confidently mis-attributed number is not, and
this corpus is entirely numbers.

Every linearized table also carries a "Columns:" line, so the header is present
in the chunk even for rows that could not be aligned - tables are atomic
(decision 4), so that context never gets separated from the rows.
"""

from __future__ import annotations

import re
from collections import Counter

# Cells that carry no information: empty, or only currency/punctuation symbols
# left behind by the HTML conversion.
_NOISE = re.compile(r"^[\s$%()’‘'\".,:;|_-]*$")

# A caption like "(In millions, except per share amounts)" states units for the
# whole table rather than naming a column.
_CAPTION = re.compile(r"^\(\s*(in|dollars|amounts)\b", re.I)


# Typographic spaces the HTML conversion leaves inside cells, e.g. the em-space
# in '$ 245,122'.
_UNICODE_SPACE = re.compile(r"[     \s]+")


def clean_cell(cell: str) -> str:
    """Normalise whitespace inside a cell and tighten currency prefixes.

    Cells arrive as '$ 245,122' - a currency symbol, an em-space, then the
    figure. Left alone that reaches the embedder as odd tokens, so it becomes
    '$245,122'. Substring matching for gold spans is unaffected either way,
    since '245,122' is present in both forms.
    """
    cell = _UNICODE_SPACE.sub(" ", cell).strip()
    return re.sub(r"^([$€£])\s+", r"\1", cell)


def compact(row: list[str]) -> list[str]:
    """Drop empty and symbol-only cells, which are HTML colspan artifacts."""
    cleaned = (clean_cell(c) for c in row)
    return [c for c in cleaned if c and not _NOISE.match(c)]


def _modal_width(rows: list[list[str]]) -> int:
    """Most common compacted width among candidate data rows."""
    widths = [len(compact(r)) for r in rows]
    widths = [w for w in widths if w >= 2]
    return Counter(widths).most_common(1)[0][0] if widths else 0


def _header_candidates(rows: list[list[str]]) -> list[tuple[int, str | None, list[str]]]:
    """Possible header interpretations, as (rows_consumed, caption, labels).

    Three readings are offered: the first row alone, the second row alone (some
    tables lead with a units caption), and the two merged for genuine two-row
    headers.

    The merge compares the rows with their stub cell removed. Both header rows
    begin with a label for the row-name column - "(In millions)" above,
    "Year Ended June 30," below - which is not a data column. Comparing full
    widths (7 vs 4) hides the real relationship; comparing 6 vs 3 shows the
    upper row repeats once per lower-row group, so "Shares | Amount" repeated
    three times sits beneath the years 2020 | 2019 | 2018.
    """
    candidates: list[tuple[int, str | None, list[str]]] = []
    if not rows:
        return candidates

    top = compact(rows[0])
    caption, labels = _split_caption(top)
    candidates.append((1, caption, labels))

    if len(rows) > 1:
        second = compact(rows[1])
        cap2, lab2 = _split_caption(second)
        candidates.append((2, cap2, lab2))

        upper, lower = top[1:], second[1:]
        if upper and lower and len(upper) > len(lower) and len(upper) % len(lower) == 0:
            per = len(upper) // len(lower)
            merged = [f"{lower[i // per]} {upper[i]}" for i in range(len(upper))]
            candidates.append((2, caption, merged))
    return candidates


def _split_caption(columns: list[str]) -> tuple[str | None, list[str]]:
    """Separate a leading units caption from the actual column labels."""
    if columns and _CAPTION.match(columns[0]):
        return columns[0], columns[1:]
    return None, columns


def linearize_table(
    rows: list[list[str]],
    fiscal_year: int | None = None,
    section: str | None = None,
) -> str:
    """Render a table as text, attributing values to columns only when certain."""
    if not rows:
        return ""

    # Choose the header interpretation that aligns the most data rows. Ties go
    # to the reading that consumes more header rows, since a recovered two-row
    # header ("2020 Shares") is strictly more informative than the upper row
    # alone ("Shares"), which repeats and cannot be told apart.
    best: tuple[int, str | None, list[str]] | None = None
    best_score = (-1, -1)
    for consumed, caption_c, labels_c in _header_candidates(rows):
        data = rows[consumed:]
        if not labels_c or not data:
            continue
        # A header row may or may not open with a stub naming the row-label
        # column - "(In millions)" is caught as a caption, but "Year Ended
        # June 30," is not, and leaving it among the labels makes every data
        # row look one cell too narrow. Both readings are scored and the data
        # decides which is right.
        variants = [labels_c]
        if len(labels_c) > 1:
            variants.append(labels_c[1:])
        for labels_v in variants:
            target = len(labels_v) + 1  # row label + one value per column
            score = (sum(1 for r in data if len(compact(r)) == target), consumed)
            if score > best_score:
                best_score, best = score, (consumed, caption_c, labels_v)

    if best is None:
        caption0, labels0 = _split_caption(compact(rows[0]))
        best = (1, caption0, labels0)

    consumed, caption, labels = best

    lines: list[str] = []
    where = " · ".join(str(p) for p in (f"FY{fiscal_year}" if fiscal_year else None, section) if p)
    if where:
        lines.append(f"Table ({where})")
    if caption:
        lines.append(caption)
    if labels:
        lines.append("Columns: " + " | ".join(labels))

    for row in rows[consumed:]:
        cells = compact(row)
        if not cells:
            continue
        if len(cells) == 1:
            # A divider or subtotal label spanning the table, e.g. "Fiscal Year 2020".
            lines.append(cells[0])
            continue
        label, values = cells[0], cells[1:]
        if labels and len(values) == len(labels):
            pairs = ", ".join(f"{col} = {val}" for col, val in zip(labels, values))
            lines.append(f"{label}: {pairs}")
        else:
            # Width mismatch - state the values without claiming which column
            # each belongs to. The "Columns:" line above still gives context.
            lines.append(f"{label}: " + ", ".join(values))

    return "\n".join(lines)


def alignment_rate(rows: list[list[str]]) -> tuple[int, int]:
    """(aligned rows, total data rows) for the chosen header. Used by tests."""
    if not rows:
        return (0, 0)
    text = linearize_table(rows)
    aligned = sum(1 for line in text.splitlines() if " = " in line)
    total = sum(1 for r in rows if len(compact(r)) >= 2)
    return (aligned, max(total - 1, 0))
