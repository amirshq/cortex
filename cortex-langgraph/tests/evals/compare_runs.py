"""Compare golden-set evaluation runs side by side.

Reads the runs saved by eval_golden.py (tests/evals/results/<label>.json) and
prints one table: metrics as rows, runs as columns, oldest run first. The best
value in each row is marked with *, and a config section at the top shows what
each run changed (runs made before config snapshots existed show "?").

Older runs simply lack the metrics added later (re-ranker lift, latency, cost,
judge); those cells show "—".

Usage (from cortex-langgraph/):
    .venv/bin/python tests/evals/compare_runs.py                    # every run
    .venv/bin/python tests/evals/compare_runs.py baseline relevance # chosen runs, in this order
    .venv/bin/python tests/evals/compare_runs.py --md               # also write results/RESULTS.md

No API calls: it only reads JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# (row label, path into the run's summary, which direction is better: "max" | "min" | None)
# A path is a tuple of keys; alternatives separated by "|" cover renamed keys in old runs.
METRICS = [
    ("Retrieval", None, None),
    ("index coverage", ("index_coverage",), "max"),
    ("recall@1", ("recall@1",), "max"),
    ("recall@5", ("recall@5",), "max"),
    ("recall@top_k", ("recall@top_k|recall@30",), "max"),
    ("MRR@top_k", ("mrr@top_k|mrr@30",), "max"),
    ("context recall (any)", ("context_recall_any",), "max"),
    ("context recall (all)", ("context_recall_all",), "max"),
    ("Re-ranker", None, None),
    ("MRR after re-rank", ("reranker", "mrr_after"), "max"),
    ("re-ranker lift (MRR)", ("reranker", "mrr_lift"), "max"),
    ("fallback (low-confidence) rate", ("low_confidence_rate",), "min"),
    ("Answers", None, None),
    ("accuracy (keyword)", ("answer_accuracy",), "max"),
    ("refusal rate (unanswerable)", ("refusal_rate",), "max"),
    ("grounded answers (judge)", ("groundedness", "grounded_rate"), "max"),
    ("claims supported (judge)", ("groundedness", "claim_support_rate"), "max"),
    ("relevance rate (judge)", ("relevance", "relevance_rate"), "max"),
    ("relevance score (judge)", ("relevance", "relevance_score"), "max"),
    ("Accuracy by question type", None, None),
    ("fact", ("by_type", "fact", "accuracy"), "max"),
    ("numeric", ("by_type", "numeric", "accuracy"), "max"),
    ("table", ("by_type", "table", "accuracy"), "max"),
    ("multi-hop", ("by_type", "multi-hop", "accuracy"), "max"),
    ("paraphrase", ("by_type", "paraphrase", "accuracy"), "max"),
    ("Latency (s / question)", None, None),
    ("whole run, mean (all runs)", ("avg_seconds",), "min"),
    ("total, mean", ("latency", "total", "mean"), "min"),
    ("total, p95", ("latency", "total", "p95"), "min"),
    ("query LLM, mean", ("latency", "query", "mean"), "min"),
    ("embed, mean", ("latency", "embed", "mean"), "min"),
    ("search, mean", ("latency", "search", "mean"), "min"),
    ("re-rank, mean", ("latency", "rerank", "mean"), "min"),
    ("generate, mean", ("latency", "generate", "mean"), "min"),
    ("Cost", None, None),
    ("USD / question", ("cost", "avg_usd_per_question"), "min"),
    ("USD / 1,000 questions", ("cost", "usd_per_1000_questions"), "min"),
    ("LLM input tokens / question", ("cost", "avg_llm_input_tokens"), "min"),
    ("judge USD / run (eval only)", ("groundedness", "judge_usd_this_run"), None),
]

# Settings shown in the config section: (row label, path into the run's config snapshot).
# The judge is listed but never marked as a change: it grades the system, it isn't part of it.
CONFIG = [
    # The index's own record wins over config.yml: an experiment index can be
    # chunked differently from what config.yml says.
    ("chunking", ("index", "chunk_strategy")),
    ("indexed chunks", ("index", "chunks")),
    ("embedding model", ("models", "embedding", "name")),
    ("query processing", ("rag", "query_processing")),
    ("top_k", ("rag", "retrieval", "top_k")),
    ("top_n", ("rag", "retrieval", "top_n")),
    ("re-ranker", ("models", "reranker", "name")),
    ("min_score", ("rag", "reranker", "min_score")),
    ("fallback policy", ("rag", "retrieval", "fallback_policy")),
    ("RAG LLM", ("models", "rag", "name")),
    ("RAG temperature", ("models", "rag", "temperature")),
    ("judge", ("models", "judge", "name")),
]


def dig(data: dict, path: tuple):
    """Follow `path` into nested dicts; a key "a|b" tries a, then b (renamed keys)."""
    node = data
    for key in path:
        if not isinstance(node, dict):
            return None
        node = next((node[k] for k in key.split("|") if k in node), None)
    return node


def load_runs(labels: list[str]) -> list[dict]:
    if labels:
        files = [RESULTS / f"{label}.json" for label in labels]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"No such run(s) in {RESULTS}: {', '.join(missing)}")
    else:
        files = sorted(RESULTS.glob("*.json"), key=lambda f: json.loads(f.read_text()).get("created")
                       or time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f))))
    runs = []
    for f in files:
        run = json.loads(f.read_text())
        run.setdefault("label", f.stem)
        runs.append(run)
    return runs


def config_rows(runs: list[dict]) -> list[list[str]]:
    rows = []
    for label, path in CONFIG:
        values = []
        for run in runs:
            if "config" in run:
                v = dig(run["config"], path)
            elif path == ("rag", "retrieval", "top_k"):
                v = run["summary"].get("top_k", 30)  # every pre-snapshot run used 30
            elif path == ("rag", "retrieval", "top_n"):
                v = run["summary"].get("top_n", 8)
            elif path == ("models", "embedding", "name"):
                v = dig(run["summary"], ("cost", "models", "embedding")) or "?"
            elif path == ("models", "rag", "name"):
                v = dig(run["summary"], ("cost", "models", "llm")) or "?"
            elif path == ("models", "judge", "name"):
                v = dig(run["summary"], ("groundedness", "judge_model")) or "—"
            else:
                v = "?"
            if isinstance(v, float) and v.is_integer():
                v = int(v)  # 0.0 (CLI flag) and 0 (config.yml) are the same setting
            values.append("—" if v is None else str(v))
        # Mark settings that differ from the first run that recorded them, so the
        # changed knob stands out. Unknown ("?") and absent ("—") values are never compared.
        known = [v for v in values if v not in ("?", "—")]
        ref = known[0] if known else None
        rows.append([label] + [v + ("  ←" if label != "judge" and v not in ("?", "—", ref) else "")
                               for v in values])
    return rows


def metric_rows(runs: list[dict]) -> list[list[str]]:
    rows = []
    for label, path, better in METRICS:
        if path is None:
            rows.append([f"[{label}]"] + [""] * len(runs))
            continue
        values = [dig(run["summary"], path) for run in runs]
        present = [v for v in values if isinstance(v, (int, float))]
        best = (max if better == "max" else min)(present) if better and len(present) > 1 else None
        cells = []
        for v in values:
            if v is None:
                cells.append("—")
            else:
                if "lift" in label:
                    text = f"{v:+.3f}"
                elif label.startswith("USD /"):
                    text = f"{v:.6f}" if v < 0.01 else f"{v:.4f}"
                elif "tokens" in label:
                    text = f"{v:.0f}"
                else:
                    text = f"{v:.3f}".rstrip("0").rstrip(".")
                cells.append(text + (" *" if best is not None and v == best and present.count(best) < len(present) else ""))
        rows.append([label] + cells)
    return rows


def render_text(header: list[str], sections: list[tuple[str, list[list[str]]]]) -> str:
    widths = [max(len(r[i]) for _, rows in sections for r in rows + [header]) for i in range(len(header))]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(w) for h, w in zip(header, widths)), line]
    for title, rows in sections:
        out.append(title)
        out += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in rows]
        out.append(line)
    return "\n".join(out)


def render_markdown(header: list[str], sections: list[tuple[str, list[list[str]]]]) -> str:
    out = ["# Golden-set evaluation runs", "",
           "Generated by `tests/evals/compare_runs.py`; do not edit by hand.",
           "`*` = best value in the row · `←` = setting changed vs the first run · `—` = not measured in that run · "
           "`?` = run predates config snapshots.", ""]
    for title, rows in sections:
        out += [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        out += ["| " + " | ".join(c.replace("[", "**").replace("]", "**") if i == 0 and c.startswith("[") else c
                                  for i, c in enumerate(row)) + " |" for row in rows]
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="*", help="runs to compare (default: all, oldest first)")
    parser.add_argument("--md", action="store_true", help="also write results/RESULTS.md")
    args = parser.parse_args()

    runs = load_runs(args.labels)
    if not runs:
        raise SystemExit(f"No runs in {RESULTS}. Run eval_golden.py first.")
    header = ["run"] + [r["label"] for r in runs]
    sections = [("Configuration", config_rows(runs)), ("Metrics", metric_rows(runs))]

    print(render_text(header, sections))
    print("* best in row · ← setting changed vs the first run · — not measured · ? run predates config snapshots")
    if args.md:
        out = RESULTS / "RESULTS.md"
        out.write_text(render_markdown(header, sections))
        print(f"\nSaved → {out.relative_to(RESULTS.parents[2])}")


if __name__ == "__main__":
    main()
