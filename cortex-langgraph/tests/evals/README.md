# Evaluation tests

These are **not** unit tests. They call real models, cost money, need API
keys, and are non-deterministic. They are deselected by default via
`addopts = -m "not eval"` in `pytest.ini`.

```bash
# the normal suite (hermetic, no keys, no network)
pytest

# the evals
pytest -m eval -v -s

# one dimension at a time
pytest -m eval tests/evals/test_retrieval_quality.py -v -s
pytest -m eval tests/evals/test_hallucination.py -v -s
pytest -m eval tests/evals/test_latency.py -v -s
```

`-s` matters: each test prints its measured metric, and the number is the
point — the pass/fail threshold is only a floor.

## Golden-set evaluation on a real document (`eval_golden.py`)

The pytest evals above use a small fictional corpus (`data/golden_set.json`)
and pass/fail thresholds. `eval_golden.py` is a script, not a test: it runs
30 questions about a real 535-page book (`data/ai_engineering_golden.json`)
through the real RAG graph over the index you built, and reports numbers to
compare between experiments. It reports:

- index coverage;
- recall@1, recall@5 and recall@k, and MRR;
- re-ranker lift;
- latency per stage;
- cost per question;
- context recall;
- answer accuracy and refusal rate.

```bash
.venv/bin/python tests/evals/eval_golden.py --label <experiment-name>
```

Each run is saved to `results/<label>.json`, together with a snapshot of the
`models:` and `rag:` config it ran with. Runs from 4.3 on also save, per question, the chunks
the LLM saw (`results[].contexts`: text, id, position, vector score), so tools such as RAGAS
can score a run from its file. One experiment means one edit in
`src/config/config.yml` → `rag:` or `models:`, followed by a run with a new label.
Re-index first if you changed chunking or the embedding model.

To compare runs side by side (no API calls, it only reads the JSON files):

```bash
.venv/bin/python tests/evals/compare_runs.py                     # all runs, oldest first
.venv/bin/python tests/evals/compare_runs.py baseline chunk-400  # just these runs
.venv/bin/python tests/evals/compare_runs.py --md                # also write results/RESULTS.md
```

The table puts metrics in rows and runs in columns. Its markers mean:

- `*`: the best value in the row.
- `←`: a setting that differs from the first run.
- `—`: a metric that run did not measure (it predates that metric).
- `?`: a setting that run did not record (it predates config snapshots).

**Read latency with care.** Embedding and generation are API calls, so their
time varies with OpenAI's load. Identical settings have measured generation at
0.8 s in one run and 2.3 s in another. Treat a latency difference as real only
when it repeats across runs, or when it comes from a local stage: search or
re-rank.

## What each file answers

| File | Question |
|---|---|
| `test_retrieval_quality.py` | Does the right chunk come back, and does the re-ranker improve on raw vector order? |
| `test_hallucination.py` | Does the system refuse to answer what the corpus does not contain? |
| `test_latency.py` | Where does the wall-clock time actually go? |

## The corpus

`data/golden_set.json` describes a fictional company. That is deliberate:
if the model can answer a question about Veldrin Corp without retrieval,
it is fabricating, because no such facts exist in its training data. A
corpus of real facts cannot tell retrieval apart from memorisation.

Extend the JSON, not the test code, when adding cases.

## Thresholds

Thresholds are set as **regression floors, not targets** — deliberately
below current measured performance so that ordinary model drift does not
turn CI red, while a real regression still does. Tighten them as the
system improves. Every threshold is a module-level constant with a
comment explaining what breaking it would mean in production.

## The testing roadmap

This is how the RAG system is evaluated, step by step, on the golden set. Each
component is measured on its own, and a change is kept only when it wins.
Update the **Status** column as steps finish. Each run's numbers are in
`results/<label>.json`.

Legend: ✅ done · ⏳ to do (evaluator) · ❌ not started (experiment) · ⚠️ partly measured

### Phase 0: foundations

| # | Sub-step | Component | Metrics | Status |
|---|---|---|---|---|
| 0.1 | Correctness tests for every module (705 + 10 xfail, `tests/correctness/`) | All | Pass/fail: does the code *work* | ✅ |
| 0.2 | Golden set from *AI Engineering* (30 questions, `data/ai_engineering_golden.json`) | All | — (the answer key) | ✅ |
| 0.3 | Evaluator script (`eval_golden.py`) | All | Runs the golden set, saves each run | ✅ |
| 0.4 | Baseline run (`results/baseline.json`) | Ingestion, retrieval, answers | Index coverage, recall@1/5/30, MRR, context recall, accuracy, refusal rate | ✅ |
| 0.5 | Experiment: keep footnotes and captions (`results/footnotes-captions.json`) | **Ingestion (cleaning)** | Index coverage 0.75 → 0.96, accuracy 0.58 → 0.71 | ✅ (kept) |

### Phase 1: make settings configurable

| # | Sub-step | Status |
|---|---|---|
| 1.1 | Models in `src/config/config.yml` (`models:`) | ✅ |
| 1.2 | Chunking, top-k, top-n, threshold and policy in `src/config/config.yml` (`rag:`) | ✅ |

### Phase 2: extend the evaluator

These metrics are needed before some components can be evaluated properly.

| # | Sub-step | Needed for | Status |
|---|---|---|---|
| 2.1 | **Re-ranker lift**: MRR before vs after re-ranking (`results/reranker-lift.json`) | Re-ranker | ✅ MRR 0.729 → 0.774 (+0.045) |
| 2.2 | **Latency per stage**: embed, search, re-rank, generate (`results/latency-per-stage.json`) | top-k, re-ranker, LLM, vector DB | ✅ 2.7 s total; re-rank ≈ 50% |
| 2.3 | **Cost per question**: embedding + LLM tokens → $ (`results/cost-per-question.json`) | Embedding model, LLM, top-k | ✅ $0.0001 / question; LLM 99.5% |
| 2.4 | **LLM-judge groundedness**: is the answer supported by its sources? (`results/groundedness.json`) | LLM | ✅ 85% of answers grounded, 89.5% of claims supported (judge `gpt-4o`) |
| 2.5 | **LLM-judge answer relevance**: does it answer the question? (`results/relevance.json`) | LLM | ✅ 90% fully on target, 2 partial; 4 answerable questions refused |
| 2.6 | **Results table**: compare all runs side by side (`compare_runs.py` → `results/RESULTS.md`) | All experiments | ✅ |

### Phase 3: experiments, one component at a time

Each experiment changes one thing, re-runs the golden set, and keeps the change only if it wins.

| # | Component | Sub-steps (variants to try) | Main metrics | Re-index? | Status |
|---|---|---|---|---|---|
| 3.1 | **Chunking** | 3.1a size 400 / 800 / 1200 · 3.1b table-aware chunking (never cut a table from its header) (`results/chunk-*.json`) | Recall, context recall, accuracy (table questions) | Yes | ✅ winner: **800 + table_aware**. Accuracy 0.667 → 0.75, tables 2/4 → 4/4, MRR 0.729 → 0.76. 400 is cheaper and faster but loses tables (1/4); 1200 is worse on recall@1 (0.5). Not yet adopted in `config.yml` (Phase 4). |
| 3.2 | **Embedding model** | `text-embedding-3-small` (current) vs `3-large`, on the 3.1 winner (`results/embed-large.json` vs `chunk-800-tables.json`) | Recall@k, MRR, cost | Yes | ✅ winner: **3-large**. Accuracy 0.75 → 0.875, recall@30 0.833 → 0.917, paraphrase 0/4 → 2/4. Costs +17% per question and +0.2 s embedding. The re-ranker lift turns negative (−0.018), which is input for 3.4. Not yet adopted in `config.yml` (Phase 4). |
| 3.3 | **top-k / top-n** | top_k 10 / 30 / 50 × top_n 5 / 8 / 12 grid, on the 3.2 winner (`results/k*-n*.json`, control `embed-large.json`) | Recall vs latency and cost | No | ✅ winner: **top_k 10, top_n 5**. Same accuracy (0.875) and context recall as 30/8, but 1.95 s instead of 3.18 s per question (re-rank 1.6 s → 0.6 s) and 20% cheaper. top_k 50 is slower and loses q19 (0.833). top_n barely matters because the gate usually keeps fewer chunks. Not yet adopted in `config.yml` (Phase 4). |
| 3.4 | **Re-ranker** | none / MiniLM-L6 / `bge-reranker-base` (current) / `bge-reranker-v2-m3`, each with its own min_score from `sweep_reranker_threshold.py`, on the 3.3 winner (`results/rr-*.json`, control `k10-n5.json`) | Lift, fallback rate, latency | No | ✅ provisional winner: **none**. Same accuracy (0.875) and refusal (1.0) as the control, context recall (all) 0.75 → 0.833, and 1.95 s → 1.34 s. v2-m3 has the best lift (+0.04) but takes 3.5 s; MiniLM loses q20. Runner-up: base with min_score −8 (same quality, best groundedness 0.91, keeps the confidence signal). Confirm on a bigger and multi-PDF golden set; needs a "none" option in the app (Phase 4). |
| 3.5 | **Query processing** | none / rewrite / HyDE / multi-query + RRF, on the 3.4 winner (`results/qp-*.json`, control `rr-none.json`) | Recall on the **paraphrase** questions | No | ✅ winner: **none**. Every variant was worse: accuracy 0.875 → 0.79 (rewrite, multi) / 0.75 (HyDE), MRR 0.79 → 0.62–0.77, and +0.6–1.8 s per question. Rewrite and multi fix q28 but lose q29 and q30; HyDE is worst. With `3-large` the original question already matches well, and gpt-4o-mini rewrites drift away from the book's wording. |
| 3.6 | **LLM** | `gpt-4o-mini` (current) vs `gpt-4o` / `gpt-4.1-mini` / `gpt-4.1`, on the 3.5 winner (`results/llm-*.json`, control `rr-none.json`) | Accuracy, groundedness, relevance, cost, latency | No | ✅ The large models are far more grounded: 0.83 → **0.96** (`gpt-4.1`; 4 ungrounded answers → 1). Keyword accuracy 0.875 → 0.833 comes only from q20, where they correctly refuse a fact the misparsed table doesn't support. Cost ~13× ($0.15 → $2.00 per 1,000 questions), latency equal. **Choice:** `gpt-4.1` for faithfulness, `gpt-4o-mini` for cost. The judge is gpt-4o; gpt-4.1 scores the same, so self-preference doesn't explain the gap. |
| 3.7 | **Vector database** | Chroma (current) vs FAISS flat / FAISS HNSW / Qdrant local, on the same vectors with an exact reference, at 1,812 and 50,000 vectors (`bench_vector_db.py` → `results/vector_db/`) | **Search latency, memory, index size** (not quality) | Yes | ✅ winner: **keep Chroma**. Quality is identical in every database. Search is ~1 ms (book) / ~2 ms (50k), about 0.1% of a question. FAISS HNSW is fastest (0.2 ms at 50k) but approximate (top-10 overlap 0.92). Qdrant's local mode is a brute-force dev fallback (334 ms); a Qdrant server needs its own test. |

### Phase 4: lock in the results

| # | Sub-step | Status |
|---|---|---|
| 4.1 | Results table: one row per experiment, winner marked → [`EXPERIMENTS.md`](EXPERIMENTS.md) | ✅ |
| 4.2 | Put the winning settings in `config.yml` | ✅ table_aware · 3-large (new `rag_embedding` role) · top_k 10 / top_n 5 · reranker `none` (new option) · gpt-4.1 at temperature 0. App index rebuilt (1,812 chunks, 3,072 dims). |
| 4.3 | Final run with all winners combined (they may interact) | ✅ `results/final.json`: plain `config.yml` against the app's index. Reproduces `llm-gpt-4.1` exactly on retrieval (recall@1 0.75, MRR 0.793) and accuracy (0.833); 1.1–1.5 s and $2.00 per 1,000 questions; each question's chunk texts are saved for RAGAS. The new **reference** for `tests/tools/`. See `EXPERIMENTS.md` → Final run |
| 4.4 | Turn the key thresholds into regression tests, so future changes can't silently make things worse | ⏳ |

### Where each component is evaluated

| Component | Evaluated in |
|---|---|
| Ingestion (cleaning) | 0.5 ✅ |
| Retrieval (as a whole) | 0.4 ✅, then re-measured in every experiment |
| Answers | 0.4 ✅, extended by 2.4–2.5 ✅ |
| Chunking (size, tables) | **3.1** ✅ |
| Embedding model | **3.2** ✅ |
| Re-ranker | 2.1 ✅ + **3.4** ✅ |
| top-k | **3.3** ✅ |
| LLM | 2.4–2.5 ✅ + **3.6** ✅ |
| Vector database | **3.7** ✅ |
| Cost and latency per question | 2.2–2.3 ✅ |
