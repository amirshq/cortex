"""Choose a re-ranker's min_score without calling the LLM.

Each cross-encoder scores on its own scale, so a threshold tuned for one model
means nothing for another. This script retrieves the top_k candidates for every
golden question once, scores them with the given re-ranker once, and then
replays the gate for many thresholds. It reports, per threshold:

  ctx any / all   context recall: is (all) the evidence among the chunks the LLM
                  would see? Same rule as the graph: gated, sorted, cut to top_n,
                  or the vector top_n when the gate rejects everything (hybrid).
  ans rejected    answerable questions where the gate rejected EVERY chunk:
                  the answer falls back to unranked vector order (bad).
  una rejected    unanswerable questions where the gate rejected every chunk:
                  the firewall working (good).

Usage (from cortex-langgraph/):
    .venv/bin/python tests/evals/sweep_reranker_threshold.py --persist-dir data/experiments/embed-large \\
        --top-k 10 --top-n 5 --reranker-model BAAI/bge-reranker-base

Cost: one query embedding per question (fractions of a cent). No LLM calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_golden import ROOT, evidence_hits  # noqa: E402  (also loads .env and the project path)

from langchain_core.documents import Document  # noqa: E402

from src.business.core.embedding import create_embedder  # noqa: E402
from src.business.rag.re_ranker.config import ReRankerConfig  # noqa: E402
from src.business.rag.re_ranker.cross_encoder import CrossEncoderScorer  # noqa: E402
from src.business.rag.vector_store import create_vector_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persist-dir", required=True)
    parser.add_argument("--reranker-model", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--golden", default=str(ROOT / "tests/evals/data/ai_engineering_golden.json"))
    args = parser.parse_args()

    persist_dir = Path(args.persist_dir).resolve()
    model = json.loads((persist_dir / "experiment.json").read_text())["embedding_model"]
    store = create_vector_store(persist_dir=str(persist_dir), embedding=create_embedder(model=model))
    scorer = CrossEncoderScorer(ReRankerConfig(model_name=args.reranker_model))
    cases = json.loads(Path(args.golden).read_text())["cases"]

    # Retrieve and score once; every threshold below reuses these.
    scored = []
    for case in cases:
        docs = [Document(page_content=d.page_content) for d, _ in
                store.similarity_search_with_score(case["question"], k=args.top_k)]
        scores = scorer.score([(case["question"], d.page_content) for d in docs])
        scored.append((case, docs, scores))
    all_scores = sorted(s for _, _, ss in scored for s in ss)
    print(f"{args.reranker_model}: {len(all_scores)} scores, range {all_scores[0]:.2f} … {all_scores[-1]:.2f}, "
          f"median {all_scores[len(all_scores) // 2]:.2f}\n")

    lo, hi = all_scores[0], all_scores[-1]
    thresholds = sorted({round(lo + (hi - lo) * i / 20, 2) for i in range(21)} | {0.0, 0.15})
    answerable = [x for x in scored if x[0]["type"] != "unanswerable"]
    unanswerable = [x for x in scored if x[0]["type"] == "unanswerable"]
    print(f"{'min_score':>9}  {'ctx any':>7}  {'ctx all':>7}  {'ans rejected':>12}  {'una rejected':>12}")
    for t in thresholds:
        ctx_any = ctx_all = ans_rej = 0
        for case, docs, scores in answerable:
            kept = [d for d, s in sorted(zip(docs, scores), key=lambda x: -x[1]) if s >= t][: args.top_n]
            if not kept:  # hybrid fallback: the vector top_n
                ans_rej += 1
                kept = docs[: args.top_n]
            hits = evidence_hits(case, kept)
            ctx_any += any(hits)
            ctx_all += all(hits)
        una_rej = sum(all(s < t for s in scores) for _, _, scores in unanswerable)
        n = len(answerable)
        print(f"{t:>9.2f}  {ctx_any / n:>7.3f}  {ctx_all / n:>7.3f}  {ans_rej:>6}/{n:<5}  {una_rej:>6}/{len(unanswerable):<5}")


if __name__ == "__main__":
    main()
