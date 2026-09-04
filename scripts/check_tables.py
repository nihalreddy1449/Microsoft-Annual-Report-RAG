"""Verify table linearization attributes numbers to the right column.

Alignment rate alone is a misleading metric: a linearizer can align every row
and still hand 2024's revenue to 2025. These assertions come from figures read
directly out of the filings, so they fail if a column mapping ever silently
inverts.

The FY2020 case is the sharpest test in the corpus - a three-year table with
two separate percentage-change columns, exactly the shape that breaks naive
positional mapping.

Run:  python scripts/check_tables.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking.tables import compact, linearize_table  # noqa: E402

PROCESSED = ROOT / "data" / "processed"

# (fiscal year, a value identifying the table, expected (column, value) pairs)
# Values may carry a currency prefix, so the check allows an optional one
# rather than demanding an exact string.
EXPECTED = [
    (2025, "281,724", [("2025", "281,724"), ("2024", "245,122")]),
    (2024, "245,122", [("2024", "245,122"), ("2023", "211,915")]),
    (2023, "211,915", [("2023", "211,915"), ("2022", "198,270")]),
    (2022, "198,270", [("2022", "198,270"), ("2021", "168,088")]),
    (2021, "168,088", [("2021", "168,088"), ("2020", "143,015")]),
    # Three years plus two percentage-change columns.
    (2020, "143,015", [("2020", "143,015"), ("2019", "125,843"), ("2018", "110,360")]),
]


def find_tables(fiscal_year: int, marker: str) -> list[list[list[str]]]:
    """Every table containing the marker.

    A figure like 143,015 appears in several tables per report - the MD&A
    summary, the five-year financial highlights, the income statement. The
    check passes if any one of them attributes the columns correctly, since
    they are all legitimate sources for the same fact.
    """
    doc = json.loads((PROCESSED / f"FY{fiscal_year}.json").read_text(encoding="utf-8"))
    return [
        el["rows"]
        for el in doc["elements"]
        if el["kind"] == "table"
        and el.get("rows")
        and any(marker in " ".join(row) for row in el["rows"])
    ]


def main() -> int:
    if not PROCESSED.exists():
        print("No parsed output. Run scripts/run_ingestion.py first.")
        return 1

    failures: list[str] = []

    print("--- column attribution ---")
    for fiscal_year, marker, expectations in EXPECTED:
        tables = find_tables(fiscal_year, marker)
        if not tables:
            failures.append(f"FY{fiscal_year}: no table containing {marker!r}")
            print(f"  FAIL  FY{fiscal_year}  no table containing {marker!r}")
            continue

        # A column label may carry a footnote marker, e.g. "2019 (a)".
        best_missing: list[str] | None = None
        for rows in tables:
            text = linearize_table(rows, fiscal_year)
            missing = [
                f"{col} = {val}"
                for col, val in expectations
                if not re.search(rf"{re.escape(col)}[^=]{{0,12}}= [$€£]?{re.escape(val)}\b", text)
            ]
            if best_missing is None or len(missing) < len(best_missing):
                best_missing = missing
            if not missing:
                break

        if best_missing:
            failures.append(f"FY{fiscal_year}: missing {best_missing}")
            print(f"  FAIL  FY{fiscal_year}  ({len(tables)} candidate tables) missing {best_missing}")
        else:
            shown = ", ".join(f"{c} = {v}" for c, v in expectations)
            print(f"  OK    FY{fiscal_year}  {shown}")

    # Corpus-wide alignment, reported for visibility rather than asserted.
    total = attributed = 0
    for path in sorted(PROCESSED.glob("FY*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for el in doc["elements"]:
            if el["kind"] != "table" or not el.get("rows"):
                continue
            text = linearize_table(el["rows"], doc["fiscal_year"])
            body = [ln for ln in text.splitlines() if ln.startswith(("Table (", "Columns:")) is False]
            for line in body:
                if ": " in line:
                    total += 1
                    if " = " in line:
                        attributed += 1

    print("\n--- corpus coverage ---")
    if total:
        pct = attributed / total * 100
        print(f"  rows rendered            : {total}")
        print(f"  with column attribution  : {attributed} ({pct:.1f}%)")
        print(f"  values-only (no claim)   : {total - attributed} ({100 - pct:.1f}%)")
        print("\n  Unattributed rows state their values without naming columns,")
        print("  which is deliberate: a vague row beats a wrong one, and the")
        print("  'Columns:' line stays in the chunk because tables are atomic.")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed.")
        return 1
    print("All column attributions correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
