"""Shared chunking machinery: atomic units, chunk records, and assembly.

The ablation compares three chunking strategies, and its whole value rests on
being able to say the score difference came from the strategy and nothing else.
That requires one independent variable, so everything the strategies have in
common is factored out here.

Each document is first reduced to a list of *atomic units* - the pieces no
strategy is allowed to split:

  * a table, already linearized (decision 4: tables are atomic in all arms)
  * a bullet run together with the sentence introducing it (the list_group
    extension of decision 4)
  * an ordinary paragraph

The strategies then differ only in how they *group* those units. Fixed-size
walks them in order filling a character budget, structure-aware groups them by
heading, and semantic groups them by embedding similarity. None of them can
sever a table row or strand a bullet from its lead-in, because none of them
ever sees anything smaller than a unit.

Every chunk also gets the same context header (fiscal year, section,
subsection). This is applied uniformly across all three arms, so it is a
constant rather than a confound, and it matters on this corpus: the prose says
"2025" where the question says "fiscal year 2025", and a chunk lifted out of
NOTES TO FINANCIAL STATEMENTS is unattributable without it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .tables import linearize_table_parts


@dataclass
class AtomicUnit:
    """An indivisible piece of a document."""

    index: int
    text: str
    kind: str  # "paragraph" | "table" | "list"
    section: str
    subsection: str | None
    fiscal_year: int
    source: str
    rows: list[list[str]] | None = None  # original table, kept for citation

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class Chunk:
    """A retrievable unit of text, produced by one chunking strategy."""

    chunk_id: str
    text: str
    strategy: str
    fiscal_year: int
    source: str
    section: str
    subsection: str | None
    kind: str  # "prose" | "table" | "mixed"
    unit_indices: list[int] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["char_count"] = self.char_count
        return data


def context_header(fiscal_year: int, section: str, subsection: str | None) -> str:
    """The uniform provenance line prepended to every chunk in every strategy."""
    parts = [f"FY{fiscal_year}", section]
    if subsection and subsection != section:
        parts.append(subsection)
    return " | ".join(parts)


def build_units(doc: dict) -> list[AtomicUnit]:
    """Reduce a parsed document to atomic units in reading order.

    ``doc`` is one of the ``data/processed/FY20XX.json`` payloads.
    """
    units: list[AtomicUnit] = []
    fiscal_year = doc["fiscal_year"]
    source = doc["source"]

    pending_list: list[dict] = []
    pending_group: int | None = None

    def flush_list() -> None:
        """Emit a buffered bullet run and its lead-in as a single unit."""
        nonlocal pending_list, pending_group
        if not pending_list:
            return
        first = pending_list[0]
        units.append(
            AtomicUnit(
                index=len(units),
                text="\n".join(e["text"] for e in pending_list),
                kind="list",
                section=first["section"],
                subsection=first["subsection"],
                fiscal_year=fiscal_year,
                source=source,
            )
        )
        pending_list, pending_group = [], None

    for el in doc["elements"]:
        group = el.get("list_group")

        if group is not None and el["kind"] == "paragraph":
            if pending_group is not None and group != pending_group:
                flush_list()
            pending_group = group
            pending_list.append(el)
            continue

        flush_list()

        if el["kind"] == "table" and el.get("rows"):
            # A long table becomes several units, split at row boundaries with
            # the header repeated, so nothing exceeds the embedder's window.
            for part in linearize_table_parts(el["rows"], fiscal_year, el["section"]):
                units.append(
                    AtomicUnit(
                        index=len(units),
                        text=part,
                        kind="table",
                        section=el["section"],
                        subsection=el["subsection"],
                        fiscal_year=fiscal_year,
                        source=source,
                        rows=el["rows"],
                    )
                )
        elif el["kind"] == "paragraph" and el["text"].strip():
            units.append(
                AtomicUnit(
                    index=len(units),
                    text=el["text"],
                    kind="paragraph",
                    section=el["section"],
                    subsection=el["subsection"],
                    fiscal_year=fiscal_year,
                    source=source,
                )
            )
        # Headings are not content units; their text reaches chunks through the
        # section/subsection metadata that the parser already attached to every
        # element, so it is never duplicated into the body.

    flush_list()
    return units


def assemble(units: list[AtomicUnit], strategy: str, index: int) -> Chunk | None:
    """Join a group of units into one chunk with a context header."""
    if not units:
        return None

    first = units[0]
    kinds = {u.kind for u in units}
    if kinds == {"table"}:
        kind = "table"
    elif "table" in kinds:
        kind = "mixed"
    else:
        kind = "prose"

    header = context_header(first.fiscal_year, first.section, first.subsection)
    body = "\n\n".join(u.text for u in units)

    return Chunk(
        chunk_id=f"{strategy}-FY{first.fiscal_year}-{index:04d}",
        text=f"{header}\n\n{body}",
        strategy=strategy,
        fiscal_year=first.fiscal_year,
        source=first.source,
        section=first.section,
        subsection=first.subsection,
        kind=kind,
        unit_indices=[u.index for u in units],
    )
