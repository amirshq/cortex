# 🔵 DeepEval: the regression gate

DeepEval turns quality metrics into pytest pass/fail tests. Here it's used **only** for the
regression gate: a change that makes answers worse must fail. RAGAS already covers the metrics
themselves.

## What goes in this folder (planned)

| File | Purpose | Roadmap |
|---|---|---|
| `test_regression.py` | Runs the golden set through the app's RAG pipeline. Fails if faithfulness or relevance drop below the thresholds | 4.4 |
| `thresholds.yml` | The thresholds: the 4.3 final run's numbers minus a margin | 4.4 |

Retrieval recall is gated with plain asserts on our own metrics (`evals/`), since DeepEval
can't match evidence quotes.

Runs with `pytest -m tools` (deselected by default; real API calls).

**Its pytest plugin is off** (`-p no:deepeval` in `pytest.ini`), so `deepeval test run` isn't
used. Tests call `assert_test` / metrics directly and run with plain pytest.

## Status
◐ Installed (deepeval 3.9.6, app venv). Thresholds: from the 4.3 final run (`evals/results/final.json`).
