# Microsoft Annual Report RAG

A retrieval-augmented QA system over Microsoft's annual reports for FY2020–FY2025,
built to answer questions like *"How did Intelligent Cloud revenue change between
FY2024 and FY2025, and what drove it?"* — with citations, and with a refusal when
the filings don't actually contain the answer.

The point of the project isn't the chatbot. It's the measurement. I built three
chunking strategies, two retrievers, and a reranker, then ran all of it against a
30-question evaluation set to find out which choices actually mattered. Some of
what I expected to matter didn't.

---

## Results

Four configurations, 30 questions, scored by an LLM judge that is deliberately a
different model from the generator.

| configuration | recall@5 | correct | faithful | refusal | over-refusal |
|---|---:|---:|---:|---:|---:|
| fixed chunking + hybrid | 56% | 54% | 100% | 70% | 36% |
| structure-aware + hybrid | 64% | 58% | 100% | 73% | 32% |
| semantic + hybrid | 60% | 56% | 100% | 70% | 32% |
| **semantic + hybrid + reranker** | **92%** | **75%** | **100%** | **85%** | **16%** |

Retrieval alone, measured separately, goes from 60% recall@10 with plain vector
search to 100% with hybrid retrieval plus reranking.

Full write-up, including the parts that didn't work: [`results/ablation_results.md`](results/ablation_results.md)

### Three things the numbers actually showed

**The reranker is the whole ballgame.** It moved recall@5 from 60% to 92% and
correctness from 56% to 75% — more than any chunking decision by a wide margin.
The over-refusal column shows why: better context doesn't just improve answers,
it converts "I don't know" into a real answer.

**The system never made anything up.** Faithfulness is 100% across 71 substantive
answers — the judge found no claim that wasn't supported by retrieved text. That
matters more than correctness to me. `gpt-oss-120b` has obviously read a lot about
Microsoft during training, and it never once fell back on that. A wrong answer is
recoverable; a confident fabrication isn't.

**Chunking barely mattered, which is not what I expected.** The three strategies
scored 54%, 58% and 56%. That's a one-question spread on a 25-question set, where
one question is four percentage points. I built the whole project around comparing
chunking strategies and the honest conclusion is that at this corpus size, chunking
isn't the lever. Reranking is.

---

## How it works

```
6 .docx files
    ↓  custom WordprocessingML parser          (headings from bold + centring)
4,269 atomic units                             (paragraphs, tables, bullet groups)
    ↓  three chunking strategies
~2,000 chunks each                             (mean size matched within 3%)
    ↓  bge-base-en-v1.5 → Chroma
6,315 vectors + a BM25 index
    ↓  hybrid retrieval, fused with RRF
40 candidates
    ↓  bge-reranker-base cross-encoder
top 5
    ↓  gpt-oss-120b, context-only prompt
answer + citations, or a refusal
```

**Parsing.** I planned to use `unstructured`, then measured what it actually
produced on these files: 980 elements on FY2025, and **zero** headings. These are
EDGAR HTML-to-docx conversions, so every paragraph carries a presentational style
and there isn't a single Word heading style in the corpus. `unstructured` infers
titles from those style names, so it had nothing to work with. I wrote a parser
that reads the WordprocessingML directly and recovers headings from formatting —
full bolding separates a heading from body text, and centring then separates a
real section from a bold subsection. Font size doesn't work: `INCOME STATEMENTS`
and `OVERVIEW` are both 10pt.

**Tables.** Roughly 440 tables hold most of the numbers, and the columns don't line
up with the headers — HTML colspan artifacts leave empty padding cells and put `$`
in a cell of its own. On the FY2025 income statement the header row has 10 cells
and the revenue row has 13, so naive positional mapping pairs `2025` with `$` and
can hand one year's figure to another year. I score several header interpretations
against the data and pick the one that aligns the most rows; anything that still
doesn't align is emitted with its values and no column claims at all. A vague row
is recoverable, a confidently mis-attributed number isn't.

**Why hybrid retrieval.** Asked for FY2025 revenue, dense search ranked the table
containing `281,724` *sixth*, behind prose reading "Revenue increased $36.6 billion
or 15%". Those prose hits aren't wrong — they're about revenue, in the right year.
A bare numeral just doesn't give an embedding model much to hold onto. BM25 finds
it instantly and is useless on paraphrase, so I run both and fuse by rank rather
than score. Rank fusion avoids inventing a normalisation between BM25's unbounded
scores and cosine similarity, which would quietly become the knob that decides the
results.

**Evaluation.** Ground truth is 49 verbatim quotes from the filings rather than
labelled chunk IDs, because the three chunkers produce completely different chunk
boundaries and IDs couldn't be compared across them. A question counts as retrieved
if any chunk in the top *k* contains one of its quotes. Answer quality is judged by
`qwen3.8-27b` — a different model family from the generator, so nothing grades its
own homework.

---

## Running it

Needs Python 3.13, and a free [Groq](https://console.groq.com) API key for the
generation step. Everything else runs locally.

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows

# torch first, from the CUDA index if you have an NVIDIA GPU
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

copy .env.example .env                  # then paste your Groq key into .env
python scripts/check_env.py             # verifies GPU, dependencies, key
```

Build the index (one time, ~3 minutes on a GPU):

```bash
python scripts/fetch_models.py          # bge-base + bge-reranker
python scripts/run_ingestion.py         # parse the six .docx
python scripts/run_chunking.py          # three chunk sets
python scripts/run_indexing.py          # embed into Chroma
```

Then:

```bash
python app.py                           # Gradio demo at localhost:7860
```

To reproduce the measurements:

```bash
python scripts/eval_retrieval.py        # recall@k tables, no API calls
python scripts/run_ablation.py          # the full ladder, ~240 Groq calls
```

A GPU isn't required. The cross-encoder is the one component that really wants
one: scoring 40 question/chunk pairs takes about 15 seconds per query on CPU,
which is measured. It's substantially faster on my RTX 5050, but I only ever
timed generation separately from retrieval, so I don't have a clean number for
it and won't invent one.

---

## Things worth trying in the demo

Ask **"What was Microsoft's total revenue in fiscal year 2025?"** twice, once with
the reranker checkbox on and once off. With it on you get `$281,724 million
(FY2025)`. With it off, the same question returns "I don't know based on the
provided documents". That's the over-refusal row of the results table happening in
front of you.

Then ask **"How much revenue did Microsoft Copilot generate in FY2025?"** Copilot
is discussed all over these reports, so plenty of relevant-looking context comes
back — but Microsoft never breaks out Copilot revenue, and the system refuses.
Refusing when nothing is retrieved is easy. Refusing when something relevant but
insufficient comes back is the part I actually cared about.

---

## What doesn't work

**Over-refusal is the real weakness.** Even at its best the system declines 16% of
questions it should be able to answer, and without the reranker it declines about a
third of them. It fails in the safe direction, which is the right direction for
anything touching financial data, but that's where the remaining quality is sitting.

**Multi-document questions score 38%**, against 100% on the easy tier. Questions
that need evidence from several fiscal years at once are the weakest part of the
pipeline, and span recall@5 is only 65% even in the best configuration — finding
*one* supporting quote isn't the same as finding all of them.

**The risk-factors question is answered when it should be refused.** These are
annual reports to shareholders, not full 10-Ks, so there's no Risk Factors section
— they just point the reader to the Form 10-K. Asked what risk factors Microsoft
identified, the model takes "our success is highly dependent on our ability to
attract and retain qualified employees" from a talent narrative and relabels it a
risk factor. The cross-reference it should be quoting is sitting at rank 3 in its
own context. I added an explicit instruction about relevance not being sufficiency;
it didn't fix it. I left it measured rather than tuning the prompt until that one
question passed, which would just be overfitting to my own eval set.

**25 answerable questions is a small sample.** One question is four percentage
points, so I don't treat anything under about 8 points as a real difference. Six
judge calls also errored during a network outage and are excluded from the means —
a failed API call is missing data, not a wrong answer, but exclusion can't recover
the evidence that was lost. The best row's refusal score rests on two of five
refusal questions for that reason, and isn't really comparable to the others.

---

## Repo layout

```
src/
  ingestion/     WordprocessingML parser + heading detection
  chunking/      fixed, structure-aware, semantic + table linearisation
  embedding/     bge-base-en-v1.5 wrapper
  retrieval/     Chroma vector store, BM25, RRF fusion
  reranking/     bge-reranker-base cross-encoder
  generation/    Groq client, context-only prompting
  eval/          harness + LLM judge
  pipeline.py    ties it together, one config = one ablation row
scripts/         ingestion, indexing, evaluation, environment checks
eval_data/       30 questions, 49 gold spans
results/         measured numbers and the full write-up
app.py           Gradio demo
```

## Stack

`bge-base-en-v1.5` · `bge-reranker-base` · Chroma · `rank_bm25` · Groq
(`gpt-oss-120b` generating, `qwen3.8-27b` judging) · Gradio · PyTorch 2.11 + CUDA 12.8

The parser is standard library only — `zipfile` and `xml.etree`.
