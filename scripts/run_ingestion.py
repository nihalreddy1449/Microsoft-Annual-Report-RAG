"""Parse the six annual reports and write structured elements to data/processed/.

Run:  python scripts/run_ingestion.py

Also acts as a regression check on the heading detector. The detector is the
foundation of the structure-aware arm of the ablation, so if it silently
degrades, every downstream number degrades with it. The assertions below fail
loudly instead.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.docx_parser import parse_corpus  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

# Every document must contain these top-level sections. They are the 10-K
# backbone and were verified present in all six files.
REQUIRED_SECTIONS = [
    "BUSINESS",
    "NOTES TO FINANCIAL STATEMENTS",
    "INCOME STATEMENTS",
    "BALANCE SHEETS",
    "CASH FLOWS STATEMENTS",
]

# Subsections that must stay H2. If any is promoted to H1, the detector has
# regressed and sections will absorb content that does not belong to them.
#
# COVID-19 is deliberately NOT in this list: in FY2020/FY2021 it is typeset
# exactly like a top-level section (fully bold, ALL-CAPS), so classifying it
# as H1 is faithful to the document rather than a defect.
FORBIDDEN_AS_H1 = ["OVERVIEW", "GENERAL", "INCOME TAXES", "SEGMENT RESULTS OF OPERATIONS"]

MIN_H1, MAX_H1 = 15, 40


def main() -> int:
    start = time.time()
    docs = parse_corpus(RAW_DIR)
    elapsed = time.time() - start

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    print(f"Parsed {len(docs)} documents in {elapsed:.2f}s\n")
    header = f"{'doc':<8}{'H1':>5}{'H2':>6}{'paras':>8}{'tables':>8}{'chars':>10}"
    print(header)
    print("-" * len(header))

    for doc in docs:
        counts = doc.counts()
        h1 = [e for e in doc.elements if e.level == 1]
        h2 = [e for e in doc.elements if e.level == 2]
        chars = sum(len(e.text) for e in doc.elements if e.kind == "paragraph")
        print(
            f"FY{doc.fiscal_year:<6}{len(h1):>5}{len(h2):>6}"
            f"{counts['paragraph']:>8}{counts['table']:>8}{chars:>10,}"
        )

        sections = doc.sections
        label = f"FY{doc.fiscal_year}"

        if not MIN_H1 <= len(h1) <= MAX_H1:
            failures.append(f"{label}: {len(h1)} H1 sections, expected {MIN_H1}-{MAX_H1}")

        for required in REQUIRED_SECTIONS:
            if not any(required in s for s in sections):
                failures.append(f"{label}: missing required section {required!r}")

        for forbidden in FORBIDDEN_AS_H1:
            if any(s.strip() == forbidden for s in sections):
                failures.append(f"{label}: {forbidden!r} promoted to H1 (subsection bug)")

        if counts["table"] == 0:
            failures.append(f"{label}: no tables extracted")

        # Guard against the absorption bug: a subsection misread as top-level
        # steals the rest of its parent's content. BUSINESS and MD&A are the
        # two large narrative sections, so if either collapses, attribution
        # has broken somewhere upstream.
        per_section: dict[str, int] = {}
        for el in doc.elements:
            if el.kind == "paragraph":
                per_section[el.section] = per_section.get(el.section, 0) + 1
        for name in ("BUSINESS", "MANAGEMENT"):
            size = max(
                (n for s, n in per_section.items() if s.startswith(name)),
                default=0,
            )
            if size < 50:
                failures.append(
                    f"{label}: section starting {name!r} holds only {size} paragraphs "
                    "- a subsection was probably promoted and absorbed it"
                )

        out = OUT_DIR / f"FY{doc.fiscal_year}.json"
        payload = {
            "fiscal_year": doc.fiscal_year,
            "source": doc.source,
            "sections": sections,
            "elements": [e.to_dict() for e in doc.elements],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(docs)} files to {OUT_DIR.relative_to(ROOT)}/")

    print("\n--- validation ---")
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        print(f"\n{len(failures)} check(s) failed.")
        return 1

    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
