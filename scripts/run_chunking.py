"""Run every chunking strategy over the corpus and validate the output.

Writes data/processed/chunks_<strategy>.json and reports size statistics.

The check that matters is the last one: every gold evidence span must survive
inside at least one chunk. recall@k is scored by span containment, so a span
that chunking cuts in half is unreachable for that strategy no matter how good
the retriever is - the arm would be penalised for a chunking artifact rather
than for its retrieval quality, and the ablation would be measuring the wrong
thing. Fixed-size is the arm at risk here, since it cuts mid-sentence.

Run:  python scripts/run_chunking.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking.base import build_units  # noqa: E402
from src.chunking.fixed import chunk_fixed  # noqa: E402
from src.chunking.structure_aware import chunk_structure_aware  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
QUESTIONS = ROOT / "eval_data" / "questions.json"

STRATEGIES = {
    "fixed": chunk_fixed,
    "structure_aware": chunk_structure_aware,
}

try:
    from src.chunking.semantic import chunk_semantic  # noqa: E402

    STRATEGIES["semantic"] = chunk_semantic
except ImportError:
    pass


def normalise(text: str) -> str:
    for src, dst in {
        "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " ",
    }.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PROCESSED.glob("FY*.json"))]
    if not docs:
        print("No parsed documents. Run scripts/run_ingestion.py first.")
        return 1

    units_by_doc = {d["fiscal_year"]: build_units(d) for d in docs}
    total_units = sum(len(u) for u in units_by_doc.values())
    kinds: dict[str, int] = {}
    for units in units_by_doc.values():
        for unit in units:
            kinds[unit.kind] = kinds.get(unit.kind, 0) + 1
    print(f"atomic units: {total_units}  {kinds}\n")

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    gold = [(q["id"], g["fy"], g["span"]) for q in questions for g in q["gold_evidence"]]

    header = f"{'strategy':<18}{'chunks':>8}{'mean':>8}{'median':>8}{'max':>8}{'spans':>10}"
    print(header)
    print("-" * len(header))

    failures: list[str] = []
    for name, fn in STRATEGIES.items():
        all_chunks = []
        for fiscal_year, units in units_by_doc.items():
            all_chunks.extend(fn(units))

        sizes = [c.char_count for c in all_chunks]
        by_year: dict[int, str] = {}
        for chunk in all_chunks:
            by_year[chunk.fiscal_year] = by_year.get(chunk.fiscal_year, "") + "\n" + chunk.text
        blobs = {fy: normalise(text) for fy, text in by_year.items()}

        missing = [
            (qid, fy, span) for qid, fy, span in gold if normalise(span) not in blobs.get(fy, "")
        ]
        found = len(gold) - len(missing)
        print(
            f"{name:<18}{len(all_chunks):>8}{statistics.mean(sizes):>8.0f}"
            f"{statistics.median(sizes):>8.0f}{max(sizes):>8}{found:>7}/{len(gold)}"
        )

        if missing:
            for qid, fy, span in missing:
                failures.append(f"{name}: {qid} FY{fy} span lost: {span[:64]!r}")

        out = PROCESSED / f"chunks_{name}.json"
        out.write_text(
            json.dumps([c.to_dict() for c in all_chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nwrote chunk files to {PROCESSED.relative_to(ROOT)}/")
    print("\n--- validation ---")
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        print(f"\n{len(failures)} gold span(s) unreachable after chunking.")
        return 1
    print("  Every gold span survives in all strategies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
