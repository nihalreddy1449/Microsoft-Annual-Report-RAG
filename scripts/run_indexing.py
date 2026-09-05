"""Embed every chunk set and store it in Chroma, one collection per strategy.

Run:  python scripts/run_indexing.py

Rebuilds from scratch each time so re-runs are idempotent: chroma_db/ is
derived data, regenerable from data/raw/, and is gitignored.

Ends with a retrieval probe rather than just a count. A store that indexed the
right number of chunks but returns nonsense is a silent failure.

The probe records the *rank* at which known content is found, not merely
whether it appears, because this is vector search alone and the ranks it
produces are the baseline that hybrid retrieval has to beat in step 7. That
matters most for exact figures: asked for FY2025 revenue, the embedder
happily returns "Revenue increased $36.6 billion or 15%" above the table
containing the literal 281,724, because it is matching meaning and a bare
numeral carries almost none. This is the measured argument for adding BM25,
and these numbers are what will show whether it worked.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embedding.embedder import Embedder  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
STRATEGIES = ["fixed", "structure_aware", "semantic"]

# (question, fiscal years to search, string that must appear in the top hits)
PROBES = [
    ("What was Microsoft's total revenue in fiscal year 2025?", [2025], "281,724"),
    ("How many people does Microsoft employ?", [2025], "228,000"),
    ("What drove Intelligent Cloud revenue growth?", [2025], "Intelligent Cloud"),
    ("How much did Azure revenue grow?", [2025], "34%"),
    ("What was diluted earnings per share?", [2025], "13.64"),
]

# Vector search alone is expected to rank exact figures poorly; the gate only
# checks the content is reachable at all. Ranks are reported so step 7 can be
# measured against them.
PROBE_DEPTH = 20


def main() -> int:
    embedder = Embedder()
    store = VectorStore(ROOT / "chroma_db", embedder=embedder)
    print(f"embedding on: {embedder.device}\n", flush=True)

    for strategy in STRATEGIES:
        path = PROCESSED / f"chunks_{strategy}.json"
        if not path.exists():
            print(f"missing {path.name} - run scripts/run_chunking.py first")
            return 1

        chunks = json.loads(path.read_text(encoding="utf-8"))
        print(f"{strategy}: indexing {len(chunks)} chunks", flush=True)
        started = time.time()
        store.reset(strategy)
        written = store.add_chunks(strategy, chunks)
        elapsed = time.time() - started
        print(
            f"  done in {elapsed:.1f}s ({written / elapsed:.0f} chunks/s), "
            f"collection holds {store.count(strategy)}\n",
            flush=True,
        )

    print(
        f"--- vector-only retrieval baseline (rank of known content, depth {PROBE_DEPTH}) ---",
        flush=True,
    )
    header = f"  {'probe':<20}" + "".join(f"{s:>18}" for s in STRATEGIES)
    print(header)
    print("  " + "-" * (len(header) - 2))

    unreachable: list[str] = []
    ranks: dict[str, list[int]] = {s: [] for s in STRATEGIES}

    for question, years, expected in PROBES:
        cells = []
        for strategy in STRATEGIES:
            hits = store.query(strategy, question, k=PROBE_DEPTH, fiscal_years=years)
            rank = next(
                (i + 1 for i, h in enumerate(hits) if expected.lower() in h.text.lower()), None
            )
            if rank is None:
                unreachable.append(f"{strategy}: {expected!r} not in top {PROBE_DEPTH}")
                cells.append("MISSING")
            else:
                ranks[strategy].append(rank)
                cells.append(f"rank {rank}")
        print(f"  {expected:<20}" + "".join(f"{c:>18}" for c in cells), flush=True)

    print()
    for strategy in STRATEGIES:
        found = ranks[strategy]
        if found:
            top5 = sum(1 for r in found if r <= 5)
            print(
                f"  {strategy:<18} mean rank {sum(found) / len(found):4.1f}   "
                f"in top-5: {top5}/{len(PROBES)}"
            )

    print("\n  These are vector-only. Exact figures rank poorly here by nature;")
    print("  step 7 (BM25 + RRF) should lift them, and this is the baseline for it.")

    if unreachable:
        print(f"\n{len(unreachable)} probe(s) unreachable entirely:")
        for item in unreachable:
            print(f"  FAIL  {item}")
        return 1
    print("\nAll probes reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
