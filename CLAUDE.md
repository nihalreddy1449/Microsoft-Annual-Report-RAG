# Microsoft Annual Report RAG — Project Handoff
**Generated:** 2026-09-03
**Source:** Handoff from a Claude.ai planning chat, generated for Claude Code

> **How to use this file:** Save it as `CLAUDE.md` at the root of the `ms-annual-report-rag` repo. Claude Code auto-loads any file with this exact name into context at the start of every session in this directory, so this only needs to be placed once. If you'd rather not save it, pasting the contents as your opening message works too.

## TL;DR

This is a Retrieval-Augmented Generation (RAG) system built over Microsoft's own Annual Reports for fiscal years 2020–2025 (six `.docx` files), built by Nihal — a B.Tech CSE (AI) student — specifically to strengthen his resume/portfolio. The full architecture has been decided end-to-end across 9 build steps (parsing → chunking → embeddings → vector store → retrieval → reranking → generation → evaluation → interface), with only deployment left genuinely open. The repo scaffold already exists on disk and the 6 raw files are in place — but **no code has been written yet**. This is a pure planning-to-build handoff.

- **Main goal:** a resume-differentiating RAG project anchored on a defensible, *measured* ablation study — not just a working demo.
- **Current phase:** Step 0 (environment, git, GitHub) and Step 1 (ingestion) are **built and verified**. Eval set written. Next up is chunking.
- **Immediate next step:** implement the three chunking strategies (`src/chunking/`) over `data/processed/*.json`.
- **Reversed decision — do not re-propose:** `unstructured` is **no longer** the parser. It returns zero headings on this corpus; see §3 Parsing. Nothing in the pipeline imports it and it is not in `requirements.txt`.
- **Biggest pitfall to avoid:** the corpus went through several reversals (MS MARCO → power-systems/IEEE papers → Wikipedia → engineering blogs → **Microsoft Annual Reports, FINAL**). Do not suggest reverting to or re-litigating any earlier corpus option.
- **Second pitfall:** don't fabricate or reuse the "71% → 86% → 91%" example numbers from a chart Nihal liked — that was a *style* reference for how to present an ablation table, not a target to hit. Real numbers must come from actually running the evaluation harness.

## 1. Project Overview

**What:** A RAG system answering questions over Microsoft's Annual Reports FY2020–FY2025 (6 documents), with source citations, correct refusal on unanswerable questions, and a rigorous ablation study comparing chunking and reranking strategies.

**Why:** This is a personal resume/portfolio project, not a class assignment (though the difficulty-tiered eval question format was shared by a faculty member — see Section 6) or a production system. The point is to demonstrate real RAG engineering judgment — hybrid retrieval, reranking, honest measurement, safe handling of an unanswerable case — not just "a working chatbot." Throughout planning, Nihal repeatedly prioritized things that are *defensible in an interview* over things that just look impressive on the surface.

**Success criteria** (synthesized from the whole planning conversation):
- A working end-to-end pipeline: ingest → chunk → embed → retrieve → rerank → generate.
- A genuine, measured ablation comparison (fixed vs. structure-aware vs. semantic chunking, with/without reranker) — this is the centerpiece resume artifact.
- A ~30-question evaluation set spanning 6 difficulty tiers, including at least one "unanswerable" case the system correctly refuses rather than hallucinates.
- A Gradio demo with a results tab.
- A clean, GitHub-style repo: proper structure, incremental commits, a solid README.
- Deployment is optional/deferred, not a hard requirement.

## 2. Environment & File Locations

- **OS:** Windows (all paths use `C:\Users\nihal\...`). Confirm whether Claude Code is running natively on Windows or under WSL before assuming shell command syntax.
- **Hardware:** RTX 5050 GPU, 8GB VRAM available locally — this constrains which local models are realistic (embedding model + reranker need to share this budget comfortably).
- **Current repo root:** `C:\Users\nihal\Desktop\Amrita\Project\ms-annual-report-rag\`
  - This supersedes an earlier, now-abandoned download location at `C:\Users\nihal\Desktop\Amrita\Project\RAG` — the raw files were reorganized into the structured repo below. Don't reference the old path.

**Repo structure** (already created on disk):
```
ms-annual-report-rag/
├── data/
│   ├── raw/              # original .docx files — POPULATED, see below
│   └── processed/        # parsed/chunked outputs — not yet populated
├── src/
│   ├── ingestion/        # docx_parser.py — BUILT (stdlib XML parser + heading detector)
│   ├── chunking/         # fixed / structure-aware / semantic strategies — not yet written
│   ├── embedding/        # bge-base-en-v1.5 wrapper — not yet written
│   ├── retrieval/        # BM25 + vector hybrid + RRF — not yet written
│   ├── reranking/        # bge-reranker-base (+ custom later) — not yet written
│   ├── generation/       # Groq API wrapper — not yet written
│   └── eval/             # harness: recall@k + LLM-judge — not yet written
├── eval_data/
│   └── questions.json    # ~30 tiered eval questions — NOT YET WRITTEN (see Open Questions)
├── results/
│   └── ablation_results.md / .json   # ablation comparison numbers — not yet populated
├── app.py                # Gradio interface — not yet written
├── requirements.txt      # not yet created
├── .env.example           # not yet created — needs a GROQ_API_KEY placeholder
├── .gitignore              # not yet created — exclude .env, __pycache__, venv/
└── README.md               # not yet created — write progressively as the project builds
```

**Raw data files** (confirmed present in `data/raw/`):
- `2020_Annual_Report.docx`
- `2021_Annual_Report.docx`
- `2022_Annual_Report.docx`
- `2023_Annual_Report.docx`
- `2024_Annual_Report.docx`
- `2025_AnnualReport.docx` — **note:** this one is missing the underscore between "Annual" and "Report" that the other five have. Handle this naming inconsistency during ingestion rather than assuming uniform filenames.

**Domain note:** Microsoft's fiscal year ends June 30 (e.g., FY2025 ≈ July 2024–June 2025). Keep this in mind when interpreting or answering FY-labeled questions, especially temporal/comparison ones.

**Tech stack** (all decided, none yet installed or used):

| Component | Choice |
|---|---|
| Parsing | **Custom WordprocessingML parser** (stdlib `zipfile` + `xml.etree`) — `unstructured` was evaluated and rejected, see §3 |
| Chunking | Custom: fixed-size + structure-aware (heading-based) + semantic (embedding-boundary-based) |
| Embeddings | `bge-base-en-v1.5` (BAAI), local |
| Vector store | `Chroma` (embedded, local, metadata filtering by fiscal year/section) |
| Sparse retrieval | `rank_bm25` (local) |
| Reranker | `bge-reranker-base` first (local); Nihal's own Transformer/BiGRU reranker later, as a comparison |
| Generation | Groq API, `Llama 3.3 70B`, free tier — needs `GROQ_API_KEY` in `.env`, never hardcoded/committed |
| Evaluation | recall@k + LLM-as-judge (faithfulness/correctness/refusal), inside a reusable harness |
| Interface | `Gradio` (chat tab + ablation-results tab) |
| Deployment | Deferred — Hugging Face Spaces is the leading (not committed) candidate |

## 3. Key Decisions & Rationale

**Corpus — multiple reversals, final answer is Microsoft Annual Reports only:**
- MS MARCO (from his prior reranker project) — rejected; wanted real documents, not a benchmark dataset.
- A power-systems/IEEE-papers domain (matching his own published research area) — rejected.
- Wikipedia articles — abandoned when his prior project files were lost switching laptops; on restart he wanted something more specific and tech-related than general knowledge.
- Public tech company engineering blogs (Netflix, Uber, Stripe, Airbnb, Meta, etc.) — briefly locked in with a full pipeline plan built around it, then he proposed switching to Microsoft's own filings instead.
- **FINAL: Microsoft's Annual Reports only, FY2020–FY2025 (6 documents), `.docx`.** Chosen because it enables genuine temporal/multi-document reasoning and forces real table/numeric-data handling (financial statements) — a more rigorous single-company story than the blogs idea, even though it's a harder build (small, repetitive corpus; heavy tables). **Do not suggest reverting to any earlier corpus option.**

**Parsing — REVERSED on 2026-09-04, do not re-propose `unstructured`.** It was originally chosen for element-aware, reading-order-preserved output (titles/narrative/tables/lists as distinct types). Measured against the actual files, it cannot deliver the one thing it was chosen for. On FY2025 it returns **980 elements — 458 NarrativeText, 270 Text, 100 ListItem, 69 Table, and zero `Title`** — in 59 seconds per document. It infers `Title` largely from Word style names, and these six files are EDGAR HTML-to-docx conversions whose every paragraph carries a presentational style (`NormalWeb`, `la2`, `rrdsinglerule`); there is no heading style anywhere in the corpus. With no headings, the structure-aware arm of the ablation cannot be built on it.

Replaced by a **custom WordprocessingML parser** (`src/ingestion/docx_parser.py`, stdlib only) that recovers headings from the formatting the filing agent did leave behind. Full bolding separates a heading from body text; centring then separates a true section from a bold subsection. Font size does **not** discriminate — `INCOME STATEMENTS` (H1) and `OVERVIEW` (H2) are both `sz=20`. Yields 19–20 sections and ~200 subsections per document, all six parsed in **1.05s**. The interview-defensible framing is the measurement, not the library: *"I evaluated `unstructured`, measured that it returns zero headings on EDGAR conversions, and built a formatting-based detector instead."*

**Chunking — the deliberate centerpiece of the project.** A 3-way ablation (inspired by a chart Nihal liked showing "Fixed 71% → Semantic 86% → +Reranker 91%," used only as a *presentation style* reference):
1. Fixed-size chunking (baseline, ignores structure)
2. Structure-aware chunking (heading/section-based — e.g. "Risk Factors," "MD&A," "Item 8 Financial Statements")
3. Semantic chunking (embedding-based topic-boundary detection)

A reranker is then added on top of whichever performs best, producing a 4th data point. **The actual percentages must come from real measurements via the evaluation harness — never fabricate or reuse the example numbers.**

**Embeddings:** `bge-base-en-v1.5` over `bge-small-en-v1.5` (better quality, still fits 8GB VRAM comfortably) and over cloud embeddings (would break the local-first preference for no real benefit at this corpus size).

**Vector store:** `Chroma` over raw `FAISS` (need metadata filtering by fiscal year for temporal/multi-doc questions) and over `Qdrant` (real server/Docker overhead not justified at this scale — a possible future swap purely for the resume line, not a functional need).

**Retrieval:** Hybrid BM25 (`rank_bm25`) + vector search (Chroma), fused via Reciprocal Rank Fusion — financial documents are full of exact terms/numbers (fiscal years, dollar figures, product names like "Azure") that pure vector search handles poorly. Nihal has direct hands-on BM25 experience from his prior MS MARCO project, so this was a confident, fast choice for him.

**Reranking — deliberately sequenced to de-risk the timeline.** Build and validate the full pipeline with the off-the-shelf `bge-reranker-base` cross-encoder **first**. Only after that works end-to-end should his own Transformer/BiGRU reranker (adapted from the MS MARCO project) be added as a comparison. **Nihal was explicit: do not start the custom reranker before the simpler version is built and tested.**

**Generation — went through a back-and-forth:**
- Local via Ollama (Llama 3.1 8B / Qwen2.5 7B) — considered, not chosen; would compete with embeddings + reranker for the same 8GB VRAM, and quality would likely be weaker on the Hard/Multi-document/Temporal eval tiers.
- Claude API — the initial recommendation (best quality, frees the GPU entirely); Nihal accepted the "cloud" framing but specifically asked for a *free* API instead of a paid one.
- **FINAL: Groq API, `Llama 3.3 70B`, free tier.** Chosen for generous free-tier limits and genuinely fast inference (an honestly-reportable low-latency story).

**Evaluation:** Combines recall@k (retrieval quality) with LLM-as-judge scoring (faithfulness, correctness, and appropriate refusal on unanswerable questions). Nihal explicitly confirmed this "recall@k + LLM-judge" combination over retrieval-metrics-alone, after being walked through why retrieval-only metrics would miss the hallucination/refusal goals he cared about. All of it runs through a reusable **evaluation harness**: given a pipeline configuration (e.g. "structure-aware chunking, no reranker"), it runs the full eval set and produces one row of a results table — this is what actually generates the ablation comparison numbers.

**Interface:** `Gradio` over Streamlit (similar tradeoffs, but Gradio's chat interface is a more natural fit and more recognized in the ML/AI demo space specifically) and over a custom HTML/React frontend (more polished, but not worth the time tradeoff against the actual RAG engineering work). A second tab should show the ablation results interactively, not just as a README screenshot.

**Deployment:** Explicitly deferred until the core project works. Hugging Face Spaces is the current leading candidate (free, purpose-built for Gradio, ML-community-recognized) over Render/Railway (more generic, more aggressive free-tier cold starts) or skipping deployment (loses the "click and try it" credibility that matters for a resume project) — **not locked, revisit later.**

**Workflow:** Explicitly "GitHub style" — proper repo structure (already created), incremental/meaningful commits per step rather than one giant end-of-project commit, optional branch-per-experiment for the ablation variants, secrets via `.env` (never hardcoded/committed), and a README written progressively.

## 3a. Locked Methodology Decisions (approved 2026-09-04)

Six decisions raised from evidence found in the actual files, and approved via Nihal's planning chat. These are **locked** — implement them, don't re-litigate them.

1. **Structure-aware chunking uses the custom heading detector** (bold + ALL-CAPS + centred, excluding table-internal paragraphs). Forced by the absence of heading styles. **Implemented.**
2. **recall@k is scored by span containment, not gold chunk IDs.** Each eval question carries verbatim gold spans; a retrieval hits if any retrieved chunk contains any span. This is the only scheme that stays comparable across three chunkers with different boundaries. **Implemented in the eval set.**
3. **Tables are linearized row-wise into self-describing sentences** for embedding ("FY2025 Total revenue: $281,724 million"), with the original table kept in metadata for display and citation. Do not embed HTML or markdown markup. *Pending — Step 4.*
4. **Tables are atomic in all three chunking strategies** — never split. Only prose chunking varies between arms, so the ablation has exactly one independent variable. *Pending — Step 4.*
   - **Extended to lists.** These documents have no Word list numbering (`<w:numPr>` appears zero times); bullets are literal characters, one per paragraph, so an item cannot be split mid-item. But a bullet is meaningless without its lead-in ("Growth of the AI PC category." under "The Windows operating system is designed to..."). The parser therefore tags each bullet run **and its introducing sentence** with a shared `list_group` id. Chunkers must keep a group whole.
5. **The LLM judge must be a different Groq model than the generator**, to avoid self-preference bias. `.env` carries `GROQ_MODEL` and `GROQ_JUDGE_MODEL` separately. *Pending — Step 11.*
6. **Eval questions stay within FY2020–FY2025.** The faculty tier examples referenced FY2026, which does not exist in the corpus; only the Unanswerable tier may reference it. **Implemented.**

**Known constraint for Step 4 (table linearization).** Table headers and data rows do **not** align positionally. HTML `colspan` artifacts leave empty padding cells and put `$` in its own cell, so on the FY2025 income statement the header row has 10 cells while the `Revenue` row has 13, and naive positional mapping pairs `'2025'` with `'$'`. Dropping empty cells and bare currency symbols before aligning fixes the important cases (measured: 76.5% of 1,385 data rows align cleanly), but **23.5% remain ragged** and include genuine two-row headers — e.g. the FY2020 stockholders' equity table, where row 0 is `Shares | Amount | Shares | Amount | Shares | Amount` and the years `2020 | 2019 | 2018` appear in row 1 *below* it. Spot-check linearized output against the source before trusting any number.

## 4. Problems Encountered & Solutions

The laptop-switch file loss that triggered the corpus restart is now mitigated: the repo is on GitHub (private, `nihalreddy1449/Microsoft-Annual-Report-RAG`) with the six `.docx` tracked.

Technical problems hit and solved so far:

- **`unstructured` returns zero headings** — the parser reversal in §3. Root cause: no Word heading styles in EDGAR HTML-to-docx conversions.
- **Blackwell GPU needs a specific torch build** — the RTX 5050 is `sm_120`. A default PyPI wheel can install cleanly, report `cuda.is_available() == True`, and fail on the first kernel launch. Fixed by installing from the CUDA 12.8 index. `scripts/check_env.py` verifies by forcing a real GPU matmul, not by trusting `is_available()`.
- **`langdetect` installed as metadata only** — a cached wheel produced a `.dist-info` with no package beside it, so pip reported it satisfied while `import langdetect` failed. `--no-cache-dir` fixed it. Moot now that `unstructured` is gone, but the failure mode is worth recognising.
- **Heading detector false starts** — "no explicit `<w:sz>`" as the H1 rule lost every financial statement; merging any two adjacent H1s welded `FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA` onto `INCOME STATEMENTS`. Both caught by the validation harness, not by inspection.
- **A live API key was pasted into `.env.example`** (the committed template). Caught before any commit existed; key rotated, `.env` verified absent from the remote.

## 5. Current Status

- **Decisions:** Build steps 1–9 locked (§3), plus six methodology decisions (§3a). Step 10 (deployment) deferred.
- **Environment:** venv on Python 3.13.14; torch 2.11.0+cu128 verified on `sm_120` with a real GPU matmul; all dependencies importing. `scripts/check_env.py` reports "Environment ready."
- **Git/GitHub:** repo initialised, pushed to a **private** GitHub repo. `.env` gitignored and verified absent from the remote; `.env.example` holds placeholders only.
- **Step 1 ingestion — BUILT AND VERIFIED.** `src/ingestion/docx_parser.py` parses all six files in ~1.05s into `data/processed/FY20XX.json`: 19–20 H1 sections, ~200 H2 subsections, 69–76 tables per document. Two guards: `scripts/run_ingestion.py` (required sections present, no subsection promoted to H1, BUSINESS/MD&A keep their bulk) and `scripts/check_gold_spans.py` (**49/49** eval gold spans still findable after parsing).
- **Eval question set — WRITTEN.** `eval_data/questions.json`: 30 questions, exactly 5 per tier, 25 answerable + 5 refusal, 49 verbatim gold spans, every span validated against the corpus and every unanswerable question validated in reverse.
- **Not yet written:** chunking, embedding, vector store, retrieval, reranking, generation, eval harness, Gradio app, README.

## 6. Terminology & Conventions

- **"Ablation ladder"** — the planned 4-point comparison: Fixed chunking → Structure-aware chunking → Semantic chunking → best chunking + reranker. The project's centerpiece resume artifact.
- **The "faculty question tiers"** — a 6-tier difficulty framework for the eval set, shared with Nihal by a faculty member, with one example per tier already given:
  - *Easy:* "What was Microsoft's revenue in fiscal year 2025?"
  - *Medium:* "What factors contributed to Microsoft's increase in operating income in FY2025?"
  - *Hard:* "How did Microsoft's Intelligent Cloud revenue change between FY2024 and FY2025, and what were the primary drivers?"
  - *Multi-document:* "How has Microsoft's AI strategy evolved between FY2023 and FY2026?"
  - *Temporal:* "What did Microsoft report about Azure growth in FY2024, and how does that compare with FY2026?"
  - *Unanswerable:* "What percentage of Azure's 2026 revenue came specifically from ChatGPT?" — the system should refuse to invent a number rather than hallucinate one.

  Plan is ~5 questions per tier, ~30 total (not yet written — see Open Questions).
- **The "Instagram reel checklist"** — an informal production-RAG-concerns list Nihal found and liked. Split into **core** items (already part of the locked plan: reranking, context-only answers with citations, explicit "I don't know" refusal, matching embedding models on query/doc sides) and **stretch** items (P95 latency measurement, semantic caching for repeat queries, full observability/tracing of retrieval scores/tokens/latency/faithfulness/cost). Core items should not be skipped; stretch items are lower priority.

## 7. User Preferences

- Wants to understand the *why* behind every technical choice, not just the *how*.
- Prefers decisions made one at a time: 2–3 options with honest pros/cons and a clear recommendation, he chooses, then move on. This worked well throughout planning and is worth continuing for any remaining implementation-level choices.
- Comfortable admitting low familiarity with a topic and deferring to a recommendation — meet this with patience and clear explanation, not as a gap to route around. He's not a total beginner, though: real hands-on experience with BM25 and a Transformer/BiGRU reranker (MS MARCO project) and a published paper on ML-based power quality classification. Explain things — don't dumb them down.
- Strong preference for local/offline components, but willing to use free-tier cloud where the tradeoff is clearly explained (exactly how the Groq decision was reached).
- Cost-conscious about paid APIs specifically — when a paid cloud LLM was suggested, he asked for a free alternative instead.
- Cares a lot about resume/interview defensibility: real measured comparisons (never fabricated), reasoning he can explain to an interviewer, features that show engineering maturity (refusal behavior, hybrid retrieval, evaluation rigor) over surface polish.
- Explicit risk-management instinct: build and validate the simpler version of something before attempting the more ambitious/custom version (stated specifically about the reranker; worth applying generally).
- Wants proper GitHub-style development: real structure, incremental meaningful commits, progressive README — not a single end-of-project dump.

## 8. Next Steps

1. ~~Initialize git; make an initial commit of the scaffold + raw data files.~~ **DONE** — pushed to a private GitHub repo.
2. ~~Create `requirements.txt`, `.env.example`, `.gitignore`.~~ **DONE** — plus `requirements.lock.txt` for reproducibility and `scripts/check_env.py` for GPU/dependency verification.
3. ~~Implement docx ingestion/parsing (`src/ingestion/`).~~ **DONE** — custom stdlib parser, not `unstructured` (§3). Filename inconsistency handled by matching the year rather than special-casing `2025_AnnualReport.docx`. Output in `data/processed/FY20XX.json` with reading order, H1/H2 section path, fiscal-year metadata, raw table rows, and `list_group` ids.
3b. ~~Write the ~30-question eval set.~~ **DONE** — `eval_data/questions.json`, span-validated both directions.
4. **NEXT — implement the three chunking strategies** (`src/chunking/`): fixed-size, structure-aware, semantic — swappable/comparable, not hardwired to one. Must honour decisions 3, 4 and the table-alignment constraint in §3a.
5. Implement embedding generation (`src/embedding/`) with `bge-base-en-v1.5`.
6. Implement Chroma ingestion — store chunk embeddings with metadata (fiscal year, section, source file).
7. Implement hybrid retrieval (`src/retrieval/`): BM25 (`rank_bm25`) + Chroma vector search, fused via Reciprocal Rank Fusion.
8. Implement reranking (`src/reranking/`) with `bge-reranker-base` on top of hybrid retrieval results.
9. Implement generation (`src/generation/`): Groq API wrapper, prompted to answer only from retrieved context, cite sources, and explicitly refuse when nothing relevant is retrieved.
10. **Write the actual ~30-question eval set** across the 6 tiers — not yet done, needed before the harness can produce real numbers.
11. Build the evaluation harness (`src/eval/`): given a pipeline configuration, run the eval set, compute recall@k + LLM-as-judge scores, output one results row.
12. Run the harness across the ablation ladder (Fixed → Structure-aware → Semantic → best + Reranker) and record the real results.
13. Build the Gradio app (`app.py`): chat tab with citations, second tab visualizing ablation results.
14. Write `README.md` progressively as each piece lands.
15. *(Stretch, after the above)* Adapt Nihal's own Transformer/BiGRU reranker as a second reranking option; compare against `bge-reranker-base`.
16. *(Stretch, after the above)* Remaining "Instagram reel checklist" items — semantic caching, P95 latency measurement, fuller observability — only if time allows.
17. *(Deferred, revisit once the core system works)* Choose a deployment path — Hugging Face Spaces is the leading candidate.

## 9. Open Questions

**Resolved** (kept for the record, do not reopen):
- ~~The ~30 eval questions need writing.~~ Written and validated; 5 per tier confirmed.
- ~~Windows vs. WSL.~~ **Native Windows.** PowerShell and Git Bash both available — each takes its own syntax. Use **absolute paths**; the working directory drifts between tool calls. Avoid nested quotes in `python -c` from PowerShell, it mangles them; put anything non-trivial in a script file.

**Still open:**
- **Nihal should review `eval_data/questions.json`** — Claude Code drafted it; a read-through is worthwhile before the harness locks around its schema.
- **FY2025 supplies 21 of 49 gold spans.** A retriever biased toward the most recent year would score slightly high. Not distorting enough to pre-empt, but it is the first thing to rebalance if the ablation arms come out suspiciously flat.
- How far to pursue the "stretch" items (custom reranker comparison, semantic caching, full observability, P95 latency) is a matter of available time — optional, lower priority than the core ablation/eval work.
- Deployment path (Hugging Face Spaces leading candidate) — still deferred until the core system works.
