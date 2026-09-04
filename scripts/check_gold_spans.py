"""Confirm every gold evidence span survives ingestion.

recall@k is scored by span containment (see eval_data/questions.json), so a
span that the parser drops or mangles becomes permanently unfindable and every
ablation arm is penalised equally and silently. This is the join between
step 1 and step 11, and it should be re-run whenever the parser changes.

Run:  python scripts/check_gold_spans.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
QUESTIONS = ROOT / "eval_data" / "questions.json"


def normalise(text: str) -> str:
    """Fold typographic variants so matching is about content, not glyphs."""
    replacements = {
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "\xa0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip().lower()


def load_corpus() -> dict[int, str]:
    """One searchable blob per fiscal year: prose plus flattened table cells."""
    corpus: dict[int, str] = {}
    for path in sorted(PROCESSED.glob("FY*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        parts: list[str] = []
        for el in doc["elements"]:
            if el["kind"] == "table" and el.get("rows"):
                parts.extend(" ".join(row) for row in el["rows"])
            elif el["text"]:
                parts.append(el["text"])
        corpus[doc["fiscal_year"]] = normalise(" \n ".join(parts))
    return corpus


def main() -> int:
    if not PROCESSED.exists() or not list(PROCESSED.glob("FY*.json")):
        print("No parsed output found. Run scripts/run_ingestion.py first.")
        return 1

    corpus = load_corpus()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]

    checked = 0
    missing: list[tuple[str, int, str]] = []
    for question in questions:
        for gold in question["gold_evidence"]:
            checked += 1
            year = gold["fy"]
            if year not in corpus:
                missing.append((question["id"], year, "FY not parsed"))
            elif normalise(gold["span"]) not in corpus[year]:
                missing.append((question["id"], year, gold["span"]))

    print(f"corpus years parsed : {sorted(corpus)}")
    print(f"gold spans checked  : {checked}")

    if missing:
        print(f"\n{len(missing)} span(s) NOT found after ingestion:")
        for qid, year, span in missing:
            print(f"  {qid}  FY{year}  {span[:76]!r}")
        print("\nThe parser is dropping content the eval set depends on.")
        return 1

    print("\nAll gold spans are still findable in the parsed output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
