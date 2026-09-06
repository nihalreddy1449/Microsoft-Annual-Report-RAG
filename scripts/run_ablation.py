"""Run the ablation ladder and produce the results table.

Four configurations, one row each:

  1. fixed            + hybrid              the naive-chunking baseline
  2. structure_aware  + hybrid              chunk on the document's own headings
  3. semantic         + hybrid              chunk where the topic changes
  4. semantic         + hybrid + reranker   best chunking, plus the cross-encoder

Retrieval is held constant at hybrid across the first three so the only thing
changing is the chunking strategy. The fourth adds the reranker on top of the
best chunker, which is the step that isolates what reranking is worth.

Roughly 240 Groq calls against a free tier, so every question is checkpointed
as it completes and a re-run resumes rather than repeating work.

Run:  python scripts/run_ablation.py
      python scripts/run_ablation.py --fresh     (ignore checkpoints)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embedding.embedder import Embedder  # noqa: E402
from src.eval.harness import EvalHarness  # noqa: E402
from src.eval.judge import Judge  # noqa: E402
from src.generation.generator import Generator  # noqa: E402
from src.pipeline import PipelineConfig, RAGPipeline  # noqa: E402
from src.reranking.reranker import Reranker  # noqa: E402

QUESTIONS = ROOT / "eval_data" / "questions.json"
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "runs"

LADDER = [
    PipelineConfig(strategy="fixed", use_bm25=True, use_reranker=False, top_k=5),
    PipelineConfig(strategy="structure_aware", use_bm25=True, use_reranker=False, top_k=5),
    PipelineConfig(strategy="semantic", use_bm25=True, use_reranker=False, top_k=5),
    PipelineConfig(strategy="semantic", use_bm25=True, use_reranker=True, top_k=5),
]


def main() -> int:
    fresh = "--fresh" in sys.argv
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    print(f"{len(questions)} questions "
          f"({sum(1 for q in questions if q['answerable'])} answerable, "
          f"{sum(1 for q in questions if not q['answerable'])} refusal)\n", flush=True)

    # Shared across configurations: reloading bge or the cross-encoder per row
    # would dominate runtime and refill the GPU for no reason.
    embedder = Embedder()
    reranker = Reranker()
    generator = Generator()
    judge = Judge()
    print(f"generator: {generator.model}")
    print(f"judge:     {judge.model}   (deliberately different, decision 5)\n", flush=True)

    harness = EvalHarness(
        questions,
        CHECKPOINTS,
        judge=judge,
        on_progress=lambda msg: print(msg, flush=True),
    )

    summaries = []
    for i, config in enumerate(LADDER, start=1):
        print(f"[{i}/{len(LADDER)}] {config.label}", flush=True)
        started = time.time()
        pipeline = RAGPipeline(
            config, embedder=embedder, reranker=reranker, generator=generator
        )
        summary, _ = harness.run(config, pipeline=pipeline, resume=not fresh)
        summaries.append(summary)
        print(f"  done in {time.time() - started:.0f}s\n", flush=True)

    # ---- table ----
    print("=" * 96)
    print("ABLATION LADDER")
    print("=" * 96)
    header = (
        f"{'configuration':<32}{'recall@5':>10}{'correct':>9}"
        f"{'faithful':>10}{'refusal':>9}{'over-ref':>10}{'latency':>10}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        # faithful is over substantive answers only; a refusal asserts nothing
        # and would otherwise score a free 100%.
        print(
            f"{s.label:<32}"
            f"{s.recall_at_k * 100:>9.0f}%"
            f"{s.correctness * 100:>8.0f}%"
            f"{s.faithfulness * 100:>9.0f}%"
            f"{s.refusal_accuracy * 100:>8.0f}%"
            f"{s.over_refusal_rate * 100:>9.0f}%"
            f"{s.mean_latency_ms:>9.0f}ms"
        )
    counts = ", ".join(f"{s.substantive_answers}/25" for s in summaries)
    print(f"\n  faithful = over substantive answers only ({counts})")
    print("  over-ref = refused a question that WAS answerable (lower is better)")

    errors = sum(s.judge_errors for s in summaries)
    if errors:
        print(
            f"\n  NOTE: {errors} judge call(s) errored and are EXCLUDED from the means "
            "above, not scored zero - a failed API call is missing data, not a wrong answer."
        )

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "ablation_results.json"
    out.write_text(
        json.dumps(
            {
                "generator": generator.model,
                "judge": judge.model,
                "questions": len(questions),
                "note": "recall@5 is span containment; correctness/faithfulness/refusal are LLM-judged",
                "rows": [s.to_dict() for s in summaries],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
