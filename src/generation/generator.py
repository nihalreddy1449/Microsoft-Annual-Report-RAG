"""Answer generation over retrieved context, via the Groq API.

Three behaviours matter more here than answer fluency, because they are what
separate a RAG system from a chatbot that happens to have read some documents:

**Context-only answering.** The model must answer from the retrieved chunks and
nothing else. Llama-class models have read a great deal about Microsoft, so
without an explicit instruction they will happily supply a revenue figure from
memory - which may even be correct, and is therefore worse, because it looks
like the retrieval worked when it did not.

**Citations.** Every claim carries the fiscal year and section it came from, so
an answer can be checked against the filing rather than trusted.

**Refusal.** When the context does not contain the answer, the model must say
so. Five of the thirty eval questions exist purely to test this, and they are
built so that plausible-looking chunks *are* retrieved - asking for Copilot
revenue pulls up plenty of Copilot text, none of which states a revenue figure.
Refusing when retrieval returns nothing is easy; refusing when it returns
something relevant-but-insufficient is the real test.

Model ids come from .env rather than the source, because Groq's served models
change: this project was planned around llama-3.3-70b-versatile, which the API
no longer offers at all.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

DEFAULT_MODEL = "openai/gpt-oss-120b"
REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = """You answer questions about Microsoft's annual reports using ONLY the \
context provided below. You are a careful financial analyst.

Rules:
1. Use ONLY facts stated in the context. Never use prior knowledge about Microsoft, \
even if you are confident it is correct.
2. Cite the fiscal year for every figure, e.g. "revenue was $281,724 million (FY2025)".
3. If the context does not contain enough information to answer, reply with exactly: \
"{refusal}" and nothing else. Do not guess, estimate, extrapolate between years, or \
substitute a related figure for the one asked about.
4. If the question asks about a fiscal year not present in the context, refuse rather \
than answering about a different year.
5. Relevance is not sufficiency. Context that discusses the same subject without \
stating the specific fact, figure, or list asked for is NOT an answer. Do not assemble \
one by generalising from related commentary, and do not relabel ordinary business \
narrative as though it were the formal disclosure the question named.
6. Be concise. Answer the question asked, without preamble.""".format(refusal=REFUSAL)


@dataclass
class Answer:
    """A generated answer plus what it was built from."""

    text: str
    question: str
    chunk_ids: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def refused(self) -> bool:
        """True when the model declined to answer from the given context.

        Matched loosely, for two reasons. Models phrase refusal differently
        even at temperature 0, so an exact-string test would score a correct
        refusal as a hallucination. And a *redirect* counts as a refusal here:
        for the risk-factors question the ideal answer is that these reports
        contain no such section and refer the reader to the Form 10-K, which
        is a correct response rather than an evasion.

        This remains a cheap heuristic. Whether a refusal was *appropriate* is
        a judgement the LLM judge makes in step 11; string matching cannot tell
        a well-grounded redirect from a dodge, which is part of why locked
        decision 5 calls for a judge at all.
        """
        stripped = self.text.strip().lower().rstrip(".")
        patterns = (
            "i don't know",
            "i do not know",
            "not contain enough information",
            "cannot be determined from the provided",
            "context does not contain",
            "does not contain",
            "do not contain",
            "not present in the provided",
            "refer to the form 10-k",
            "refers the reader to",
            "refers readers to",
        )
        return any(p in stripped for p in patterns)


def format_context(hits: Sequence[Any], max_chars: int = 12000) -> tuple[str, list[Any]]:
    """Render retrieved chunks as a numbered context block.

    Chunks are numbered so the model can refer to them, and truncated as whole
    chunks rather than mid-text: a half-table would let the model read a value
    out of a row whose header has been cut away.
    """
    parts: list[str] = []
    used: list[Any] = []
    total = 0
    for i, hit in enumerate(hits, start=1):
        block = f"[{i}] {hit.text}"
        if total + len(block) > max_chars and used:
            break
        parts.append(block)
        used.append(hit)
        total += len(block)
    return "\n\n".join(parts), used


class Generator:
    """Groq-backed answer generation with context-only prompting."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 900,
        # Sized for an unreliable connection rather than an ideal one. Measured
        # here: 2 of 8 probes to the API failed, and the failures arrive in
        # bursts lasting tens of seconds rather than as independent drops. Five
        # retries span only ~60s of backoff, which a single burst outlasted and
        # killed a run. Eight, with the wait capped at 60s, tolerates roughly
        # four minutes of intermittent connectivity.
        max_retries: int = 8,
        backoff_cap: int = 60,
        env_path: str | Path | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_cap = backoff_cap
        if api_key is None or model is None:
            from dotenv import load_dotenv

            load_dotenv(env_path or Path(__file__).resolve().parents[2] / ".env")

        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
        self.model = model or os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_MODEL
        self.temperature = temperature
        # gpt-oss models reason before answering, and reasoning tokens come out
        # of this budget. Too small a value returns an empty answer rather than
        # an error - measured: 120 tokens produced a blank reply.
        self.max_tokens = max_tokens
        self._client = None

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add it.")

    @property
    def client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def _complete(self, question: str, context: str):
        """One completion, retrying transient failures with backoff.

        A full ablation is ~120 generation calls over roughly half an hour, so
        transient failures are expected rather than exceptional. A dropped DNS
        lookup killed a run at this exact point once, wasting a completed
        configuration's worth of API calls, because retries had been given to
        the judge and not to the generator.

        Rate limits and connection errors are retried; an authentication or
        bad-request failure is raised immediately, since repeating it would
        only waste time and quota.
        """
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {question}",
                        },
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                if any(fatal in name for fatal in ("Authentication", "PermissionDenied", "BadRequest")):
                    raise
                last = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, self.backoff_cap))
        raise RuntimeError(f"generation failed after {self.max_retries} attempts: {last}")

    def answer(self, question: str, hits: Sequence[Any]) -> Answer:
        """Answer a question from retrieved chunks."""
        if not hits:
            # Nothing retrieved: refuse without spending an API call.
            return Answer(text=REFUSAL, question=question, model=self.model)

        context, used = format_context(hits)
        started = time.time()
        response = self._complete(question, context)
        latency = (time.time() - started) * 1000

        text = (response.choices[0].message.content or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        usage = response.usage

        return Answer(
            text=text or REFUSAL,
            question=question,
            chunk_ids=[h.chunk_id for h in used],
            fiscal_years=sorted({int(h.metadata["fiscal_year"]) for h in used}),
            sections=sorted({str(h.metadata.get("section", "")) for h in used}),
            model=self.model,
            latency_ms=latency,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
