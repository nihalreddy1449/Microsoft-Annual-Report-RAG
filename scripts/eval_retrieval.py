"""Measure retrieval quality: recall@k for every chunking strategy and retriever.

This is the first real ablation measurement. It scores retrieval only - no
generation, no LLM judge - so it isolates whether the right evidence reaches
the model at all.

Scoring follows locked decision 2: a question counts as retrieved if any chunk
in the top k contains any of its verbatim gold spans. Two numbers are reported:

  recall@k       fraction of questions where at least one gold span was found.
                 The headline metric.
  span recall@k  fraction of all 49 gold spans found. Lower and more demanding,
                 and the honest metric for the multi-document and temporal
                 tiers, where answering properly needs evidence from several
                 documents rather than any one of them.

**No fiscal-year filtering.** The gold data records which year each span comes
from, and passing that to the retriever would leak the answer - the system does
not know the year at question time. Chroma's year filter exists for the
application, not for this measurement.

The five unanswerable questions carry no gold spans and are excluded here;
refusal behaviour is a generation property, measured by the judge in step 11.

Run:  python scripts/eval_retrieval.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embedding.embedder import Embedder  # noqa: E402
from src.retrieval.bm25 import BM25Index  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
QUESTIONS = ROOT / "eval_data" / "questions.json"
RESULTS = ROOT / "results"

STRATEGIES = ["fixed", "structure_aware", "semantic"]
RETRIEVERS = ["vector", "bm25", "hybrid"]
K_VALUES = [1, 3, 5, 10]


def normalise(text: str) -> str:
    for src, dst in {
        "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " ",
    }.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    answerable = [q for q in questions if q["answerable"]]
    total_spans = sum(len(q["gold_evidence"]) for q in answerable)
    print(f"{len(answerable)} answerable questions, {total_spans} gold spans\n", flush=True)

    embedder = Embedder()
    store = VectorStore(ROOT / "chroma_db", embedder=embedder)

    results: dict[str, dict[str, dict[int, dict[str, float]]]] = {}

    for strategy in STRATEGIES:
        chunks = json.loads((PROCESSED / f"chunks_{strategy}.json").read_text(encoding="utf-8"))
        started = time.time()
        bm25 = BM25Index(chunks)
        hybrid = HybridRetriever(store, bm25, strategy)
        print(f"{strategy}: {len(chunks)} chunks, BM25 built in {time.time() - started:.1f}s",
              flush=True)

        results[strategy] = {}
        max_k = max(K_VALUES)

        for retriever in RETRIEVERS:
            hits_at: dict[int, int] = {k: 0 for k in K_VALUES}
            spans_at: dict[int, int] = {k: 0 for k in K_VALUES}

            for question in answerable:
                if retriever == "vector":
                    found = store.query(strategy, question["question"], k=max_k)
                elif retriever == "bm25":
                    found = bm25.search(question["question"], k=max_k)
                else:
                    found = hybrid.search(question["question"], k=max_k)

                texts = [normalise(h.text) for h in found]
                spans = [normalise(g["span"]) for g in question["gold_evidence"]]

                for k in K_VALUES:
                    window = texts[:k]
                    matched = sum(1 for s in spans if any(s in t for t in window))
                    if matched:
                        hits_at[k] += 1
                    spans_at[k] += matched

            results[strategy][retriever] = {
                k: {
                    "recall": hits_at[k] / len(answerable),
                    "span_recall": spans_at[k] / total_spans,
                }
                for k in K_VALUES
            }

    # ---- report ----
    print("\n" + "=" * 74)
    print("recall@k  (question is a hit if any gold span appears in the top k)")
    print("=" * 74)
    header = f"{'strategy':<18}{'retriever':<12}" + "".join(f"{'@' + str(k):>9}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for strategy in STRATEGIES:
        for retriever in RETRIEVERS:
            row = "".join(
                f"{results[strategy][retriever][k]['recall'] * 100:>8.0f}%" for k in K_VALUES
            )
            print(f"{strategy:<18}{retriever:<12}{row}")
        print()

    print("=" * 74)
    print(f"span recall@k  (fraction of all {total_spans} gold spans found)")
    print("=" * 74)
    print(header)
    print("-" * len(header))
    for strategy in STRATEGIES:
        for retriever in RETRIEVERS:
            row = "".join(
                f"{results[strategy][retriever][k]['span_recall'] * 100:>8.0f}%" for k in K_VALUES
            )
            print(f"{strategy:<18}{retriever:<12}{row}")
        print()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "retrieval_results.json"
    out.write_text(
        json.dumps(
            {
                "questions": len(answerable),
                "gold_spans": total_spans,
                "note": "retrieval only; no fiscal-year filter, which would leak the answer",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
