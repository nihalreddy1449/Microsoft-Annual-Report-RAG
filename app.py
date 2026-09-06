"""Gradio demo: ask questions over Microsoft's annual reports, and see the ablation.

Two tabs, because the project has two things worth showing:

  **Ask** - the working system, with the retrieved context shown alongside every
  answer. Showing sources is the point rather than decoration: a RAG answer you
  cannot trace back to a document is indistinguishable from a fluent guess, and
  the pipeline's measured strength is that it does not guess.

  **Results** - the measured ablation, read from results/ablation_results.json
  so the page cannot drift from what was actually measured.

The retrieval settings are exposed deliberately. Switching the reranker off and
re-asking the same question is the fastest way to see what the ablation table
is claiming - usually the answer degrades into a refusal, which is exactly the
over-refusal effect the numbers record.

Run:  python app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr

from src.embedding.embedder import Embedder
from src.generation.generator import Generator
from src.pipeline import PipelineConfig, RAGPipeline
from src.reranking.reranker import Reranker

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "ablation_results.json"

STRATEGIES = ["semantic", "structure_aware", "fixed"]

# Built once and shared by every pipeline. Loading bge (419MB) and the
# cross-encoder (1.1GB) per configuration would make switching settings
# unusable, and on CPU it would be far worse.
_shared: dict[str, object] = {}
_pipelines: dict[tuple, RAGPipeline] = {}


def get_pipeline(strategy: str, use_bm25: bool, use_reranker: bool, top_k: int) -> RAGPipeline:
    """Return a pipeline for this configuration, reusing loaded models."""
    key = (strategy, use_bm25, use_reranker, int(top_k))
    if key in _pipelines:
        return _pipelines[key]

    if "embedder" not in _shared:
        _shared["embedder"] = Embedder()
        _shared["generator"] = Generator()
    if use_reranker and "reranker" not in _shared:
        _shared["reranker"] = Reranker()

    config = PipelineConfig(
        strategy=strategy,
        use_bm25=use_bm25,
        use_reranker=use_reranker,
        top_k=int(top_k),
    )
    _pipelines[key] = RAGPipeline(
        config,
        embedder=_shared["embedder"],  # type: ignore[arg-type]
        reranker=_shared.get("reranker"),  # type: ignore[arg-type]
        generator=_shared["generator"],  # type: ignore[arg-type]
    )
    return _pipelines[key]


def format_sources(hits) -> str:
    """Render retrieved chunks so a claim can be traced to a filing."""
    if not hits:
        return "_No context retrieved._"

    lines = ["### Retrieved context\n"]
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        year = meta.get("fiscal_year", "?")
        section = str(meta.get("section", ""))[:60]
        subsection = str(meta.get("subsection", "") or "")[:50]
        where = f"FY{year} · {section}" + (f" · {subsection}" if subsection else "")

        score = getattr(hit, "score", None)
        found_by = getattr(hit, "found_by", None)
        badge = f" · {found_by}" if found_by else ""
        score_text = f" · score {score:.3f}" if isinstance(score, float) else ""

        body = hit.text
        # Strip the context header we prepend at chunk time; it is already
        # shown in the source line above and would just be noise here.
        if "\n\n" in body:
            body = body.split("\n\n", 1)[1]
        body = body.strip()
        if len(body) > 700:
            body = body[:700] + " …"

        lines.append(f"**[{i}] {where}**{score_text}{badge}\n\n> {body}\n")
    return "\n".join(lines)


def answer(message: str, history: list, strategy: str, use_bm25: bool,
           use_reranker: bool, top_k: int):
    """Handle one turn: retrieve, generate, and show the sources behind it."""
    message = (message or "").strip()
    if not message:
        return history, "", "_Ask a question to see the retrieved context._", ""

    history = list(history) + [{"role": "user", "content": message}]

    try:
        pipeline = get_pipeline(strategy, use_bm25, use_reranker, top_k)
        started = time.time()
        hits = pipeline.retrieve(message)
        retrieval_ms = (time.time() - started) * 1000
        result = pipeline.generator.answer(message, hits)
    except Exception as exc:  # noqa: BLE001
        note = (
            f"**{type(exc).__name__}**: {exc}\n\n"
            "If this mentions the API key, check `.env` has a valid `GROQ_API_KEY`."
        )
        history.append({"role": "assistant", "content": f"Something went wrong.\n\n{note}"})
        return history, "", "_No context retrieved._", ""

    history.append({"role": "assistant", "content": result.text})

    years = ", ".join(f"FY{y}" for y in result.fiscal_years) or "none"
    stats = (
        f"retrieval {retrieval_ms:.0f} ms · generation {result.latency_ms:.0f} ms · "
        f"{result.prompt_tokens}+{result.completion_tokens} tokens · "
        f"context from {years}"
        + ("  ·  **refused**" if result.refused else "")
    )
    return history, "", format_sources(hits), stats


def load_results() -> tuple[list[list], str]:
    """Read the measured ablation table from disk."""
    if not RESULTS.exists():
        return [], "_No results yet. Run `python scripts/run_ablation.py`._"

    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = []
    for row in data.get("rows", []):
        rows.append([
            row["label"],
            f"{row['recall_at_k'] * 100:.0f}%",
            f"{row['correctness'] * 100:.0f}%",
            f"{row['faithfulness'] * 100:.0f}%",
            f"{row['refusal_accuracy'] * 100:.0f}%",
            f"{row['over_refusal_rate'] * 100:.0f}%",
            f"{row['mean_latency_ms']:.0f} ms",
        ])
    caption = (
        f"Generator **{data.get('generator', '?')}**, judged by "
        f"**{data.get('judge', '?')}** — deliberately a different model family, "
        "so the generator never grades its own output."
    )
    return rows, caption


FINDINGS = """
### What these numbers say

**Reranking is the lever.** recall@5 goes 60% → 92% and correctness 56% → 75%.
The over-refusal column shows the mechanism: better context does not merely
improve answers, it converts refusals into answers.

**The system fails safe.** Faithfulness is 100% across 71 substantive answers —
no claim unsupported by the retrieved context. The generator has certainly read
about Microsoft in training, and never once answered from memory. For a
financial assistant that is the property that matters: a wrong answer is
recoverable, a confident fabrication is not.

**Chunking is not the story.** The three chunkers score 54% / 58% / 56%
correctness — a one-question spread on a 25-question set. At this corpus size,
chunking is not where the quality is.

### What they do not say

- **25 answerable questions is a small sample.** One question is four
  percentage points. Differences under ~8 points are not real.
- **The best row's refusal score rests on two questions, not five.** Three
  refusal questions hit judge errors during a network outage and are excluded —
  a failed API call is missing data, not a wrong answer, but exclusion cannot
  recover the evidence.
- **Multi-document questions score 38%**, against 100% on easy ones. They need
  evidence from several fiscal years at once, and that is the weakest part of
  the pipeline.

Full write-up, including the bugs found while auditing these numbers:
`results/ablation_results.md`.
"""


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _rerank_info() -> str:
    """Set expectations honestly - on CPU this is the slow step, by a lot.

    Measured on this machine with the GPU disabled: ~15s per query warm, since
    the cross-encoder scores 40 question/chunk pairs one at a time through a
    278M model. On GPU it is well under a second. Worth stating rather than
    leaving the user to wonder whether the app has hung.
    """
    if _device() == "cuda":
        return "The single biggest quality gain. Sub-second on this GPU."
    return (
        "The single biggest quality gain, and the slow step on CPU "
        "(~15s/query here). Turn it off to see the difference - answers often "
        "degrade into refusals, which is the over-refusal effect in the results tab."
    )


def build() -> gr.Blocks:
    with gr.Blocks(title="Microsoft Annual Report RAG", fill_height=True) as demo:
        device_note = (
            "Running on **GPU**."
            if _device() == "cuda"
            else "Running on **CPU** — reranking takes ~15s per query here; it is "
            "sub-second on a GPU. The first query also loads ~1.5GB of models."
        )
        gr.Markdown(
            "# Microsoft Annual Report RAG\n"
            "Question answering over Microsoft's annual reports, FY2020–FY2025. "
            "Answers come only from retrieved text, with sources shown, and the "
            "system refuses when the filings do not contain an answer.\n\n"
            f"_{device_note}_"
        )

        with gr.Tabs():
            with gr.Tab("Ask"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chat = gr.Chatbot(
                            label="Conversation",
                            height=420,
                            layout="panel",
                        )
                        question = gr.Textbox(
                            placeholder="e.g. How did Intelligent Cloud revenue change between FY2024 and FY2025?",
                            label="Question",
                            lines=2,
                        )
                        with gr.Row():
                            send = gr.Button("Ask", variant="primary")
                            clear = gr.Button("Clear")
                        stats = gr.Markdown("")

                        gr.Examples(
                            examples=[
                                "What was Microsoft's total revenue in fiscal year 2025?",
                                "How did Intelligent Cloud revenue change between FY2024 and FY2025?",
                                "How has Microsoft's AI strategy evolved between FY2023 and FY2025?",
                                "What did Microsoft report about Azure growth in FY2020 versus FY2025?",
                                "How much revenue did Microsoft Copilot generate in FY2025?",
                            ],
                            inputs=question,
                            label="Try one (the last should be refused - Copilot revenue is never disclosed)",
                        )

                    with gr.Column(scale=2):
                        with gr.Accordion("Retrieval settings", open=True):
                            strategy = gr.Dropdown(
                                STRATEGIES,
                                value="semantic",
                                label="Chunking strategy",
                            )
                            use_bm25 = gr.Checkbox(
                                value=True,
                                label="Hybrid retrieval (BM25 + vector)",
                                info="Off = vector only. Exact figures rank poorly without BM25.",
                            )
                            use_reranker = gr.Checkbox(
                                value=True,
                                label="Cross-encoder reranking",
                                info=_rerank_info(),
                            )
                            top_k = gr.Slider(
                                1, 10, value=5, step=1,
                                label="Chunks passed to the model",
                            )
                        sources = gr.Markdown(
                            "_Ask a question to see the retrieved context._",
                            label="Sources",
                        )

            with gr.Tab("Ablation results"):
                rows, caption = load_results()
                gr.Markdown("## Measured ablation\n" + caption)
                gr.Dataframe(
                    value=rows,
                    headers=["configuration", "recall@5", "correct", "faithful",
                             "refusal", "over-refusal", "latency"],
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown(FINDINGS)

        inputs = [question, chat, strategy, use_bm25, use_reranker, top_k]
        outputs = [chat, question, sources, stats]
        send.click(answer, inputs=inputs, outputs=outputs)
        question.submit(answer, inputs=inputs, outputs=outputs)
        clear.click(
            lambda: ([], "", "_Ask a question to see the retrieved context._", ""),
            outputs=outputs,
        )

    return demo


if __name__ == "__main__":
    build().launch()
