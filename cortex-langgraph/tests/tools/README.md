# tests/tools: third-party evaluation and monitoring tools

The `tests/` folder answers three different questions, one folder each:

| Folder | Question | Built with | Cost |
|---|---|---|---|
| `correctness/` | Does the code **work**? | 🛠️ pytest + fakes | Free, no network |
| `evals/` | How **good** are the results? | 🛠️ our own evaluator (`eval_golden.py`), golden set, experiments | Cents per run |
| **`tools/`** | The same questions, answered with **standard tools**, plus a **live view** of the app | 🟠 Langfuse · 🟢 RAGAS · 🔵 DeepEval · 📈 Prometheus/Grafana | Cents per run; Langfuse runs in Docker |

`evals/` stays the **reference**. The tools here build on its output (for example, RAGAS scores
the `evals/results/<label>.json` files) and are checked against it before they are trusted.
Nothing in `evals/` or `correctness/` depends on this folder.

## One subfolder per tool

| Folder | Tool | Role in Cortex | Roadmap steps |
|---|---|---|---|
| [`langfuse/`](langfuse/README.md) | 🟠 Langfuse | The dashboard for LLM activity: traces, live latency and cost, scores, user feedback, datasets | 5.2, 5.4, 5.5, 5.7 |
| [`ragas/`](ragas/README.md) | 🟢 RAGAS | Standard quality metrics (context precision, faithfulness, relevance) and test-set generation | 2.7, 4.5, 4.7, 5.3 |
| [`deepeval/`](deepeval/README.md) | 🔵 DeepEval | The regression gate: fail when quality drops | 4.4 |
| [`grafana/`](grafana/README.md) | 📈 Prometheus + Grafana | System health, alerts, knowledge-base monitoring | 5.1, 5.6 |

LangSmith is not used. It does the same job as Langfuse, and Langfuse is open source,
self-hosted (the data stays on this machine) and can trace cortex-core too.

## Rules

- **Each tool stays in its own subfolder.** Code shared by several tools (for example,
  loading a results file) goes in a module at this level, not inside one tool's folder.
- **The app never imports from here.** The one exception is Langfuse tracing, which has to
  hook into the running graphs: that hook lives in `src/`, is off by default, and is switched on
  from `.env`. Everything else about Langfuse (Docker setup, datasets, scripts) lives here.
- **Two environments.** Langfuse and DeepEval need the app, so they're installed into the
  app's `.venv` from [`requirements.txt`](requirements.txt). RAGAS conflicts with the app's
  packages (it would downgrade openai and langchain-openai), and only reads results files, so
  it has **its own venv** in `ragas/.venv`, from [`ragas/requirements.txt`](ragas/requirements.txt).
  Neither is in the app's `requirements.txt`.
- **DeepEval's pytest plugin is off** (`-p no:deepeval` in `pytest.ini`). It imports deepeval
  when pytest starts, and deepeval rejects the empty provider settings in `.env`, which crashed
  every test run. `conftest.py` drops those empty settings before a test imports deepeval.
- **Tests here are opt-in.** `conftest.py` marks them `tools`, and `pytest.ini` deselects that
  marker, so a plain `pytest` run stays free and offline. Every test file starts with
  `pytest.importorskip(...)`.
- **Scripts run from `cortex-langgraph/`**, like the ones in `evals/`.

```bash
# once
.venv/bin/pip install -r tests/tools/requirements.txt                    # Langfuse + DeepEval
python3.11 -m venv tests/tools/ragas/.venv
tests/tools/ragas/.venv/bin/pip install -r tests/tools/ragas/requirements.txt   # RAGAS

.venv/bin/python -m pytest -m tools                                       # the tool tests (e.g. the DeepEval gate)
tests/tools/ragas/.venv/bin/python tests/tools/ragas/<script>.py          # RAGAS scripts
```

## Status

| Tool | Status |
|---|---|
| 🟠 Langfuse | ◐ SDK 4.15.6 (app venv); server running on http://localhost:3002 and verified; app tracing hook to do |
| 🟢 RAGAS | ◐ installed (0.4.3, own venv); scripts to do |
| 🔵 DeepEval | ◐ installed (3.9.6, app venv); regression tests to do |
| 📈 Grafana | ◐ the stack runs from `Docker/`; knowledge-base panels to do |
