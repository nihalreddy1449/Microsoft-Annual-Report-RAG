"""LLM-as-judge scoring for generated answers.

Locked decision 5: the judge must be a different model from the generator.
Self-preference bias is a documented effect - models rate their own output more
generously - and since the whole point of this project is a defensible
measurement, a generator grading its own homework would undermine it. Here the
generator is openai/gpt-oss-120b and the judge is qwen/qwen3.8-27b, from a
different family entirely rather than merely a different size.

Three things are scored, because they fail independently:

  **correctness**   does the answer match the known-correct answer?
  **faithfulness**  is every claim supported by the retrieved context, or was
                    something invented? An answer can be correct and unfaithful
                    at once - if the model recalls Microsoft's revenue from
                    training data while the retrieval failed, the answer looks
                    right and the system is broken.
  **refusal**       for the unanswerable questions, did it decline? A refusal
                    is correct only when the corpus genuinely lacks the answer;
                    refusing an answerable question is scored as a failure too.

The judge sees the gold answer for correctness, and the retrieved context for
faithfulness. It never sees which pipeline produced the answer, so it cannot
favour a configuration.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_JUDGE_MODEL = "qwen/qwen3.8-27b"

ANSWERABLE_PROMPT = """You are grading a retrieval-augmented QA system on Microsoft's \
annual reports. Be strict and literal.

QUESTION:
{question}

KNOWN-CORRECT ANSWER (ground truth):
{gold}

CONTEXT THE SYSTEM WAS GIVEN:
{context}

THE SYSTEM'S ANSWER:
{answer}

Score these independently:

correctness: judge whether the answer answers THE QUESTION, using the known-correct \
answer as the reference for the facts. The known-correct answer often carries extra \
context beyond what was asked - do not penalise an answer for leaving out detail the \
question did not request.
  2 = answers the question, and every figure it gives matches the reference.
  1 = partly answers it - a multi-part question with one part missing, or one figure \
wrong among several correct ones.
  0 = wrong, contradicts the reference, or refuses when the answer was available.

faithfulness: 1 if every factual claim in the answer is supported by the CONTEXT \
above. 0 if any figure, name or claim does not appear in the context - even if it \
happens to be true in reality. Judge only against the context, not your own knowledge.

Reply with ONLY this JSON, no other text:
{{"correctness": 0, "faithfulness": 0, "reason": "one short sentence"}}"""

REFUSAL_PROMPT = """You are grading a retrieval-augmented QA system on Microsoft's \
annual reports. This question is UNANSWERABLE from the corpus - the documents do not \
contain the information.

QUESTION:
{question}

WHY IT IS UNANSWERABLE:
{gold}

CONTEXT THE SYSTEM WAS GIVEN:
{context}

THE SYSTEM'S ANSWER:
{answer}

Score:

refusal: 1 if the system declined to answer, said it did not know, or correctly \
redirected the reader elsewhere (for example, noting the information is in a \
different document). 0 if it supplied an answer, estimate, or figure - including \
one assembled by generalising from loosely related material in the context.

Reply with ONLY this JSON, no other text:
{{"refusal": 0, "reason": "one short sentence"}}"""


@dataclass
class Verdict:
    """One judged answer."""

    question_id: str
    tier: str
    answerable: bool
    correctness: int | None = None  # 0-2, answerable only
    faithfulness: int | None = None  # 0-1, answerable only
    refusal: int | None = None  # 0-1, unanswerable only
    reason: str = ""
    judge_model: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model reply.

    Models wrap JSON in prose or code fences even when told not to, and some
    emit a <think> block first, so the reply is cleaned before parsing rather
    than trusted.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {text[:160]!r}")
    return json.loads(match.group(0))


class Judge:
    """Scores answers with a model deliberately different from the generator."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        # Matches the generator: both call the same host over the same
        # connection and are equally exposed to it dropping.
        max_retries: int = 8,
        backoff_cap: int = 60,
        env_path: str | Path | None = None,
    ) -> None:
        if api_key is None or model is None:
            from dotenv import load_dotenv

            load_dotenv(env_path or Path(__file__).resolve().parents[2] / ".env")

        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
        self.model = model or os.environ.get("GROQ_JUDGE_MODEL", "").strip() or DEFAULT_JUDGE_MODEL
        self.max_retries = max_retries
        self.backoff_cap = backoff_cap
        self._client = None

        generator_model = os.environ.get("GROQ_MODEL", "").strip()
        if generator_model and self.model == generator_model:
            raise RuntimeError(
                f"Judge model {self.model!r} is the same as the generator. Locked decision 5 "
                "requires them to differ, to avoid self-preference bias."
            )

    @property
    def client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def _call(self, prompt: str) -> dict:
        """One judged call, retrying through rate limits with backoff."""
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                return _extract_json(response.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                # A bad key or malformed request will not fix itself; retrying
                # only burns quota and time.
                if any(f in name for f in ("Authentication", "PermissionDenied", "BadRequest")):
                    raise
                last = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, self.backoff_cap))
        raise RuntimeError(f"judge failed after {self.max_retries} attempts: {last}")

    def judge(self, question: dict, answer_text: str, context: str) -> Verdict:
        """Score one answer against its question."""
        verdict = Verdict(
            question_id=question["id"],
            tier=question["tier"],
            answerable=question["answerable"],
            judge_model=self.model,
        )
        # Long contexts are trimmed for the judge only; the generator saw it all.
        context = context[:8000]

        try:
            if question["answerable"]:
                data = self._call(
                    ANSWERABLE_PROMPT.format(
                        question=question["question"],
                        gold=question["gold_answer"],
                        context=context,
                        answer=answer_text,
                    )
                )
                verdict.correctness = max(0, min(2, int(data.get("correctness", 0))))
                verdict.faithfulness = max(0, min(1, int(data.get("faithfulness", 0))))
            else:
                data = self._call(
                    REFUSAL_PROMPT.format(
                        question=question["question"],
                        gold=question["gold_answer"],
                        context=context,
                        answer=answer_text,
                    )
                )
                verdict.refusal = max(0, min(1, int(data.get("refusal", 0))))
            verdict.reason = str(data.get("reason", ""))[:200]
        except Exception as exc:  # noqa: BLE001
            verdict.error = f"{type(exc).__name__}: {str(exc)[:160]}"

        return verdict
