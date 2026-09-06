"""Evaluation harness: run one pipeline configuration over the full eval set.

Given a PipelineConfig, produce one row of the ablation table. That framing is
deliberate - the harness is the thing that makes the comparison reproducible,
so adding a configuration means adding a row, not writing new scoring code.

Results are checkpointed per question. About 240 API calls go into a full
ablation against a free tier, and losing a completed run to a rate limit near
the end would be painful, so each answer and verdict is written as it lands and
a re-run resumes from the checkpoint instead of repeating work.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..pipeline import PipelineConfig, RAGPipeline
from .judge import Judge, Verdict


def normalise(text: str) -> str:
    for src, dst in {
        "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " ",
    }.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class QuestionResult:
    question_id: str
    tier: str
    answerable: bool
    answer: str
    retrieved_ids: list[str] = field(default_factory=list)
    span_hit: bool = False
    spans_found: int = 0
    spans_total: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    verdict: dict[str, Any] = field(default_factory=dict)
    generation_error: str = ""

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        return data


@dataclass
class RunSummary:
    """One row of the ablation table."""

    label: str
    config: dict[str, Any]
    questions: int
    recall_at_k: float
    span_recall: float
    correctness: float  # mean of 0-2, reported as a fraction of 2
    faithfulness: float
    refusal_accuracy: float
    over_refusal_rate: float
    substantive_answers: int
    mean_latency_ms: float
    total_tokens: int
    judge_errors: int
    generation_errors: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class EvalHarness:
    """Runs the eval set through a configured pipeline and scores the output."""

    def __init__(
        self,
        questions: list[dict],
        checkpoint_dir: Path,
        judge: Judge | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self.questions = questions
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._judge = judge
        self.on_progress = on_progress or (lambda msg: None)

    @property
    def judge(self) -> Judge:
        if self._judge is None:
            self._judge = Judge()
        return self._judge

    def _checkpoint_path(self, label: str) -> Path:
        safe = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        return self.checkpoint_dir / f"{safe}.json"

    def run(
        self,
        config: PipelineConfig,
        pipeline: RAGPipeline | None = None,
        resume: bool = True,
    ) -> tuple[RunSummary, list[QuestionResult]]:
        """Run every question through one configuration and score the answers."""
        label = config.label
        checkpoint = self._checkpoint_path(label)

        done: dict[str, dict] = {}
        if resume and checkpoint.exists():
            # An entry whose judge call errored holds no score, so it is not
            # "done" - re-running must retry it rather than inherit the gap.
            cached = json.loads(checkpoint.read_text("utf-8"))
            done = {r["question_id"]: r for r in cached
                    if not r.get("verdict", {}).get("error") and not r.get("generation_error")}
            stale = len(cached) - len(done)
            if stale:
                self.on_progress(f"  re-judging {stale} question(s) that errored previously")
            self.on_progress(f"  resuming: {len(done)}/{len(self.questions)} already scored")

        pipeline = pipeline or RAGPipeline(config)
        results: list[QuestionResult] = []

        # Checkpoint state is keyed by question id and seeded with everything
        # already on disk, so a write never drops cached entries that this run
        # has not reached yet. Writing `results` alone truncated the file to
        # however far the run had got: re-judging question 1 of 30 and then
        # being interrupted left a single entry and discarded 29 scored ones.
        merged: dict[str, dict] = dict(done)

        def save() -> None:
            ordered = [merged[q["id"]] for q in self.questions if q["id"] in merged]
            checkpoint.write_text(
                json.dumps(ordered, ensure_ascii=False, indent=1), encoding="utf-8"
            )

        for i, question in enumerate(self.questions, start=1):
            if question["id"] in done:
                results.append(QuestionResult(**done[question["id"]]))
                continue

            hits = pipeline.retrieve(question["question"])

            # One question failing after all its retries should not discard a
            # run that is otherwise complete. The failure is recorded, scored
            # as zero, and surfaced in the summary, so a degraded run is
            # visibly degraded rather than silently missing.
            try:
                answer = pipeline.generator.answer(question["question"], hits)
                generation_error = ""
            except Exception as exc:  # noqa: BLE001
                from ..generation.generator import Answer

                answer = Answer(text="", question=question["question"])
                generation_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                self.on_progress(f"  [{i}/{len(self.questions)}] {question['id']} GENERATION FAILED")

            spans = [normalise(g["span"]) for g in question["gold_evidence"]]
            texts = [normalise(h.text) for h in hits]
            found = sum(1 for s in spans if any(s in t for t in texts))

            context = "\n\n".join(f"[{n}] {h.text}" for n, h in enumerate(hits, 1))
            if generation_error:
                # Nothing to judge; record the failure rather than scoring an
                # empty string as a wrong answer, which would be indistinguishable
                # from a genuine model mistake in the results.
                verdict = Verdict(
                    question_id=question["id"],
                    tier=question["tier"],
                    answerable=question["answerable"],
                    error=generation_error,
                )
            else:
                verdict = self.judge.judge(question, answer.text, context)

            result = QuestionResult(
                question_id=question["id"],
                tier=question["tier"],
                answerable=question["answerable"],
                answer=answer.text,
                retrieved_ids=[h.chunk_id for h in hits],
                span_hit=found > 0,
                spans_found=found,
                spans_total=len(spans),
                latency_ms=answer.latency_ms,
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                verdict=verdict.to_dict(),
                generation_error=generation_error,
            )
            results.append(result)
            merged[question["id"]] = result.to_dict()

            # Written every question: a rate limit near the end of a long run
            # must not cost the whole run.
            save()
            self.on_progress(f"  [{i}/{len(self.questions)}] {question['id']}")
            time.sleep(0.2)  # stay clear of free-tier request rate limits

        return self.summarise(label, config, results), results

    @staticmethod
    def summarise(label: str, config: PipelineConfig, results: list[QuestionResult]) -> RunSummary:
        answerable = [r for r in results if r.answerable]
        refusals = [r for r in results if not r.answerable]

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        # A judge call that errored is missing data, not a zero. Scoring it 0
        # silently punishes a configuration for the network being down, which
        # is exactly what happened on the first full run: four of five refusal
        # questions errored and the row reported 77% refusal accuracy when it
        # had really only scored one of them.
        scored = [r for r in answerable if not r.verdict.get("error")]
        correctness = [r.verdict["correctness"] for r in scored
                       if r.verdict.get("correctness") is not None]
        # Faithfulness is only meaningful for answers that assert something. A
        # refusal makes no factual claims, so it is trivially faithful, and
        # including refusals drove this metric to 100% in every arm - a number
        # that looked like a strength and was measuring nothing.
        substantive = [r for r in scored if not _looks_refused(r.answer)]
        faithfulness = [r.verdict["faithfulness"] for r in substantive
                        if r.verdict.get("faithfulness") is not None]
        # Refusal accuracy spans both halves: declining an unanswerable question
        # and *not* declining an answerable one are the same skill, and a system
        # that refuses everything would otherwise score perfectly.
        refusal_scores = [float(r.verdict["refusal"]) for r in refusals
                          if not r.verdict.get("error") and r.verdict.get("refusal") is not None]
        wrongly_refused = [0.0 if _looks_refused(r.answer) else 1.0 for r in answerable]

        return RunSummary(
            label=label,
            config=config.__dict__.copy(),
            questions=len(results),
            recall_at_k=mean([1.0 if r.span_hit else 0.0 for r in answerable]),
            span_recall=(
                sum(r.spans_found for r in answerable) / sum(r.spans_total for r in answerable)
                if sum(r.spans_total for r in answerable)
                else 0.0
            ),
            correctness=mean([c / 2.0 for c in correctness]),
            faithfulness=mean([float(f) for f in faithfulness]),
            refusal_accuracy=mean(refusal_scores + wrongly_refused),
            over_refusal_rate=mean(
                [1.0 if _looks_refused(r.answer) else 0.0 for r in answerable]
            ),
            substantive_answers=len([r for r in answerable if not _looks_refused(r.answer)]),
            mean_latency_ms=mean([r.latency_ms for r in results]),
            total_tokens=sum(r.prompt_tokens + r.completion_tokens for r in results),
            judge_errors=sum(1 for r in results if r.verdict.get("error")),
            generation_errors=sum(1 for r in results if r.generation_error),
        )


def _looks_refused(text: str) -> bool:
    lowered = text.strip().lower()
    return any(p in lowered for p in ("i don't know", "i do not know", "does not contain"))
