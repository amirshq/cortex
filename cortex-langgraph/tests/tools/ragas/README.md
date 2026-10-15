# 🟢 RAGAS: standard RAG metrics and test-set generation

RAGAS fills two gaps in `evals/`: **context precision** (are the chunks sent to the LLM
relevant?) and a **bigger golden set**. It also gives a second opinion on our groundedness and
relevance judges.

## What goes in this folder (planned)

| File | Purpose | Roadmap |
|---|---|---|
| `ragas_score.py` | Reads `evals/results/<label>.json` and adds context_precision, context_recall, faithfulness and answer_relevancy. Needs the retrieved chunk texts in the results file (an `eval_golden.py` change) | 2.7 |
| `meta_eval.py` | Question by question: our judge vs RAGAS. Agree / disagree, and why | 4.7 |
| `generate_testset.py` | Generates 150–200 questions from the book, converted to our golden-set format for review by hand | 4.5 |

**Judge model:** pinned to the same `models.judge` as `eval_golden.py` (gpt-4o, temperature 0),
so the scores are comparable.

**Its own venv.** RAGAS runs in `tests/tools/ragas/.venv`, not the app's `.venv`: it needs an
older openai, langchain-openai and langchain-community than the app. The scripts here only read
`evals/results/*.json` and never import `src/`. Setup is in [`requirements.txt`](requirements.txt).

## Status
◐ Installed (ragas 0.4.3, own venv, imports verified). Scripts not started.
