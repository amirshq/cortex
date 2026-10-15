"""Evaluate the RAG pipeline against a golden set.

Runs every golden question through the REAL RAG graph (vector search → re-rank →
gate → LLM) over the index that is currently built, and reports:

  Index coverage     — is the evidence in the index at all? (ingestion problems)
  Retrieval          — recall@1/@5/@top_k and MRR of vector search (top_k from config.yml)
  Re-ranker lift     — the same candidates re-ordered by the cross-encoder alone
                       (no gate, no cut): MRR / recall@1 / recall@5 before vs after
  Latency per stage  — embed (query embedding) / search (vector DB) / re-rank /
                       generate (LLM): mean, p50, p95 and share of the total
  Cost per question  — query-embedding tokens + LLM input/output tokens → USD
                       (re-ranker runs locally: $0), plus the one-time cost of
                       embedding the whole index
  Context recall     — is the evidence among the chunks the LLM actually sees
                       (after re-ranking / fallback)?
  Answers            — accuracy (must_contain_any) and refusal rate on
                       unanswerable questions
  Groundedness       — an LLM judge (config.yml → models.judge) splits each answer
                       into factual claims and checks every claim against the
                       chunks the LLM was given: % of answers fully grounded and
                       % of claims supported. Calibrated before every run.
  Relevance          — the same judge, shown only the QUESTION and the ANSWER:
                       does the answer address what was asked (answers / partial /
                       does_not_answer / refusal)? Calibrated before every run.
  Per type           — the same, split by fact / numeric / table / multi-hop / paraphrase

Evidence is matched by TEXT, not chunk id, so one golden set works for any chunk
size. A chunk "contains" a quote if it contains the whole normalised quote or any
8-word window of it (quotes may be split across chunk boundaries).

Usage (from cortex-langgraph/):
    .venv/bin/python tests/evals/eval_golden.py
    .venv/bin/python tests/evals/eval_golden.py --golden tests/evals/data/ai_engineering_golden.json --label baseline

Each run is saved to tests/evals/results/<label>.json so runs can be compared.
Costs: one embedding + one LLM call per question, plus one judge call
(about 10 cents for 30 questions with gpt-4o as judge). --no-judge skips it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # cortex-langgraph/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import tiktoken  # noqa: E402
from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from typing import Literal  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

from src.business.core.cost import calculate_chat_cost, calculate_embedding_cost  # noqa: E402
from src.business.core.embedding import create_embedder  # noqa: E402
from src.business.core.model import create_llm  # noqa: E402
from src.business.core.prompt_builder import format_context  # noqa: E402
from src.utils.config import load_config, model_settings  # noqa: E402
from src.business.rag import rag_persist_dir  # noqa: E402
from src.business.rag.vector_store import create_vector_store  # noqa: E402
from src.business.rag.re_ranker.config import ReRankerConfig  # noqa: E402
from src.business.rag.re_ranker.re_ranker import final_score  # noqa: E402
from src.business.rag.retrieval import RAGPipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_processing import MODES as QUERY_MODES, QueryProcessingStore  # noqa: E402

REFUSAL_MARKERS = ("don't know", "do not know", "does not", "doesn't", "not mention",
                   "no information", "not specified", "not provided", "not stated", "cannot")
WINDOW = 8  # words


# ── text matching ─────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """Lower-case, unify quotes/dashes, drop Markdown table/emphasis marks,
    collapse whitespace, and remove spaces before punctuation ("facts ." → "facts.")."""
    text = text.lower()
    text = (text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
                .replace("–", "-").replace("—", "-"))
    text = re.sub(r"[|*_#`]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" ([.,;:!?)])", r"\1", text)
    return text.strip()


def quote_in(quote: str, chunk_norm: str) -> bool:
    words = normalize(quote).split()
    if len(words) <= WINDOW:
        return " ".join(words) in chunk_norm
    return any(" ".join(words[i:i + WINDOW]) in chunk_norm for i in range(len(words) - WINDOW + 1))


def evidence_hits(case: dict, docs) -> list[set[int]]:
    """For each evidence quote, the positions (0-based) of the docs that contain it."""
    norms = [normalize(d.page_content) for d in docs]
    return [{i for i, n in enumerate(norms) if quote_in(ev["quote"], n)} for ev in case["evidence"]]


def first_hit(hits: list[set[int]]) -> int | None:
    """Best (0-based) position of ANY evidence quote, or None if none was found."""
    return min((min(h) for h in hits if h), default=None)


def context_record(docs) -> list[dict]:
    """The chunks the LLM was given, in prompt order, with their text.

    Saved per question so tools that score (question, contexts, answer) records, such as
    RAGAS (tests/tools/ragas/), can grade a run from its results file without re-running it.
    """
    keep = ("source_id", "section", "chunk_start", "chunk_end", "vector_score", "rerank_score")
    return [{"id": getattr(d, "id", None), "text": d.page_content,
             **{k: d.metadata[k] for k in keep if k in d.metadata}} for d in docs]


def rerank_order(pipeline: RAGPipeline, question: str, candidates):
    """The candidates re-ordered by the cross-encoder alone — no gate, no top-n cut,
    so the re-ranker's ordering is compared with vector order on the same list."""
    if not candidates or pipeline.scorer is None:  # no re-ranker configured: order unchanged
        return list(candidates)
    scores = pipeline.scorer.score([(question, d.page_content) for d in candidates])
    scored = [d.model_copy(update={"metadata": {**d.metadata, "rerank_score": float(s)}})
              for d, s in zip(candidates, scores)]
    return sorted(scored, key=lambda d: final_score(d, pipeline.reranker_config), reverse=True)


# ── latency per stage ─────────────────────────────────────────────────────────
STAGES = ("query", "embed", "search", "rerank", "generate", "total")


# text-embedding-3-* tokenise with cl100k_base; the API's own token count is
# dropped by LangChain, so it is recounted here (exact for these models).
_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


class TimedEmbeddings(Embeddings):
    """Wraps the real embedder and adds up the time and tokens spent embedding queries.

    The vector store embeds the query inside similarity_search_with_score(), so
    this is how the "retrieve" node's time is split into embed vs search."""

    def __init__(self, inner: Embeddings, model: str = ""):
        self.inner = inner
        self.model = model
        self.seconds = 0.0
        self.tokens = 0

    def embed_documents(self, texts):
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        self.tokens += count_tokens(text)
        t0 = time.perf_counter()
        try:
            return self.inner.embed_query(text)
        finally:
            self.seconds += time.perf_counter() - t0


def run_timed(pipeline: RAGPipeline, embedder: TimedEmbeddings, question: str) -> tuple[dict, dict]:
    """Run the real graph node by node and time each stage.

    stream_mode="updates" yields once per finished node, so the gap between
    yields is that node's duration. LLM tokens are collected by LangChain's
    usage callback. Returns (final state, seconds per stage, cost)."""
    embedder.seconds, embedder.tokens = 0.0, 0
    state, node_s = {"question": question}, defaultdict(float)
    with get_usage_metadata_callback() as usage:
        start = last = time.perf_counter()
        for update in pipeline.graph.stream({"question": question}, stream_mode="updates"):
            now = time.perf_counter()
            for node, delta in update.items():
                node_s[node] += now - last
                state.update(delta or {})
            last = now
    query_s = getattr(pipeline.vector_store, "llm_seconds", 0.0)  # query-processing LLM call (3.5)
    stages = {
        "query": query_s,
        "embed": embedder.seconds,
        "search": node_s["retrieve"] - embedder.seconds - query_s,
        "rerank": node_s["rerank"],
        "generate": node_s["generate"] + node_s["fallback"],  # fallback is a list slice: ~0s
        "total": last - start,
    }
    return state, {k: round(v, 3) for k, v in stages.items()}, question_cost(embedder.tokens, usage.usage_metadata, embedder.model,
                         getattr(pipeline, "cost_model", None) or model_settings("rag")["name"])


def question_cost(embed_tokens: int, llm_usage: dict, embedding_model: str, llm_model: str) -> dict:
    """Tokens and USD for one question. Prices: src/business/core/cost.py."""
    llm_in = sum(u.get("input_tokens", 0) for u in llm_usage.values())
    llm_out = sum(u.get("output_tokens", 0) for u in llm_usage.values())
    embed_usd = calculate_embedding_cost(embed_tokens, embedding_model)
    # Priced by the name we asked for: the API reports a dated one (gpt-4o-mini-2024-07-18).
    llm_usd = calculate_chat_cost(llm_model, llm_in, llm_out)
    return {"embedding_model": embedding_model, "llm_model": llm_model, "embed_tokens": embed_tokens, "llm_input_tokens": llm_in, "llm_output_tokens": llm_out,
            "embed_usd": embed_usd, "llm_usd": llm_usd, "total_usd": round(embed_usd + llm_usd, 6)}


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 3)


def is_refusal(answer: str) -> bool:
    return any(m in answer.lower() for m in REFUSAL_MARKERS)


def contains_expected(answer: str, expected: list[str]) -> bool:
    """Whole-word match: "form" must not count as found inside "information"."""
    a = normalize(answer)
    return any(re.search(rf"(?<!\w){re.escape(normalize(e))}(?!\w)", a) for e in expected)


# ── re-ranker variants ────────────────────────────────────────────────────────
class VectorOrderScorer:
    """"No re-ranker": scores that reproduce the vector store's own order, so the
    graph runs unchanged but keeps the top_n vector results (used with
    min_score=-inf, so the gate never rejects anything)."""

    def score(self, text_pairs):
        return [float(len(text_pairs) - i) for i in range(len(text_pairs))]


# ── groundedness (LLM judge) ──────────────────────────────────────────────────
JUDGE_PROMPT = """You are a strict fact-checker for a question-answering system.

You get the QUESTION, the CONTEXT the system was given, and the ANSWER it wrote.
1. Split the ANSWER into atomic factual claims (one fact each). Restating the
   QUESTION or its premise ("The author warns against X because ...") is not a
   claim; only what the ANSWER adds is.
2. For each claim, decide whether the CONTEXT supports it: stated in it, or a
   direct consequence of it. A claim that is true in the real world but NOT in
   the CONTEXT is unsupported. Rewording is fine; new numbers, names or details are not.
3. A refusal ("I don't know", "the context does not say") is not a claim:
   return no claims for it."""


class Claim(BaseModel):
    claim: str = Field(description="One atomic factual claim made by the ANSWER.")
    supported: bool = Field(description="True only if the CONTEXT supports this claim.")


class Verdict(BaseModel):
    claims: list[Claim] = Field(description="Every factual claim in the ANSWER; empty for a refusal.")


RELEVANCE_PROMPT = """You grade whether an ANSWER addresses the QUESTION it was given.
Judge ONLY whether it responds to what was asked, not whether it is true: you are
not shown any source, so do not guess at correctness.

Labels:
- answers: directly answers everything the QUESTION asks.
- partial: gives SOME of what was asked but not all of it, e.g. one half of a
  two-part question, or a range or rough value where a specific one was asked.
- does_not_answer: gives NONE of what was asked for. Background, related facts or
  a description of the topic WITHOUT the requested value, name or reason is
  does_not_answer, not partial, however on-topic it is.
- refusal: says it does not know or that the information is not available."""


class Relevance(BaseModel):
    label: Literal["answers", "partial", "does_not_answer", "refusal"]
    reason: str = Field(description="One short sentence explaining the label.")


# answers = 1, partial = 0.5, does_not_answer = 0; refusals are counted separately.
RELEVANCE_SCORE = {"answers": 1.0, "partial": 0.5, "does_not_answer": 0.0}


class Judge:
    """Grades groundedness and relevance with the model in config.yml → models.judge."""

    def __init__(self):
        self.model = model_settings("judge")["name"]
        llm = create_llm(role="judge")
        self.grounded_llm = llm.with_structured_output(Verdict)
        self.relevance_llm = llm.with_structured_output(Relevance)
        self.usd = 0.0

    def _ask(self, llm, system: str, human: str):
        with get_usage_metadata_callback() as usage:
            out = llm.invoke([("system", system), ("human", human)])
        tokens_in = sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values())
        tokens_out = sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values())
        self.usd += calculate_chat_cost(self.model, tokens_in, tokens_out)
        return out

    def relevance(self, question: str, answer: str) -> dict:
        v = self._ask(self.relevance_llm, RELEVANCE_PROMPT, f"QUESTION:\n{question}\n\nANSWER:\n{answer}")
        return {"label": v.label, "reason": v.reason}

    def grade(self, question: str, context: str, answer: str) -> dict:
        verdict = self._ask(self.grounded_llm, JUDGE_PROMPT,
                            f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}")
        claims = verdict.claims
        return {"claims": len(claims), "supported": sum(c.supported for c in claims),
                "grounded": all(c.supported for c in claims),
                "unsupported": [c.claim for c in claims if not c.supported]}

    def calibrate(self) -> None:
        """A judge that approves everything would make the score meaningless, so
        it must accept a faithful answer and reject a planted hallucination."""
        question = "How many bits does TF32 occupy?"
        context = "[1] The TF32 format occupies 19 bits, although it is named for 32."
        good = self.grade(question, context, "TF32 occupies 19 bits.")
        bad = self.grade(question, context, "TF32 occupies 19 bits and was introduced by Google in 2015.")
        if not good["grounded"] or bad["grounded"]:
            raise SystemExit(f"Judge failed calibration (faithful={good}, planted={bad}); "
                             f"its scores can't be trusted. Try another models.judge.")
        print(f"Judge {self.model}: groundedness calibration passed "
              f"(faithful answer accepted, planted claim {bad['unsupported']} rejected).")

        # Relevance: one answer per label the judge must tell apart.
        expected = {
            "TF32 occupies 19 bits.": "answers",
            "TF32 is a floating-point format used on NVIDIA GPUs.": "does_not_answer",
            "I don't know.": "refusal",
        }
        got = {a: self.relevance(question, a)["label"] for a in expected}
        two_part = self.relevance("How many tokens was Llama 3 trained on, and how many parameters does its largest model have?",
                                  "Llama 3 was trained on 15 trillion tokens.")["label"]
        if got != expected or two_part != "partial":
            raise SystemExit(f"Judge failed relevance calibration (got {got}, two-part half answer → {two_part!r}).")
        print(f"Judge {self.model}: relevance calibration passed (answers / partial / does_not_answer / refusal told apart).")


# ── evaluation ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=str(ROOT / "tests/evals/data/ai_engineering_golden.json"))
    parser.add_argument("--label", default=time.strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-judge groundedness check")
    parser.add_argument("--persist-dir", help="index to evaluate (default: the app's index). "
                        "Experiment indexes: data/experiments/<label>/, from build_experiment_index.py")
    parser.add_argument("--embedding-model", help="model for embedding the questions (default: the one recorded "
                        "in the index's experiment.json, else config.yml → models.embedding.name)")
    parser.add_argument("--top-k", type=int, help="override rag.retrieval.top_k (candidates from vector search)")
    parser.add_argument("--top-n", type=int, help="override rag.retrieval.top_n (chunks sent to the LLM)")
    parser.add_argument("--reranker-model", help="override models.reranker.name; 'none' = keep vector order (no gate)")
    parser.add_argument("--min-score", type=float, help="override rag.reranker.min_score (each model has its own scale)")
    parser.add_argument("--query-mode", choices=QUERY_MODES, default="none",
                        help="query processing before vector search: rewrite | hyde | multi (experiment 3.5)")
    parser.add_argument("--rag-model", help="override models.rag.name: the LLM that writes the answers (experiment 3.6)")
    parser.add_argument("--rag-temperature", type=float,
                        help="override models.rag.temperature for this run (0 = repeatable answers)")
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text())
    cases = golden["cases"]
    persist_dir = str(Path(args.persist_dir).resolve() if args.persist_dir else rag_persist_dir())
    # Questions must be embedded with the same model as the index, or the
    # vectors aren't comparable. An experiment index records its model.
    experiment = Path(persist_dir, "experiment.json")
    index_model = json.loads(experiment.read_text()).get("embedding_model") if experiment.exists() else None
    embedding_model = args.embedding_model or index_model or model_settings("rag_embedding")["name"]
    if index_model and embedding_model != index_model:
        raise SystemExit(f"The index was embedded with {index_model}; questions can't use {embedding_model}.")
    # Same store and embedder as production; the wrapper only adds a stopwatch.
    embedder = TimedEmbeddings(create_embedder(model=embedding_model), embedding_model)
    rag_model = args.rag_model or model_settings("rag")["name"]
    llm = (create_llm(role="rag", model_name=rag_model, temperature=args.rag_temperature)
           if args.rag_temperature is not None or args.rag_model else None)
    overrides = {k: v for k, v in (("top_k_input", args.top_k), ("top_n_output", args.top_n),
                                   ("min_score", args.min_score)) if v is not None}
    no_reranker = args.reranker_model == "none"
    if no_reranker:
        overrides["min_score"] = float("-inf")
    elif args.reranker_model:
        overrides["model_name"] = args.reranker_model
    store = create_vector_store(persist_dir=persist_dir, embedding=embedder)
    if args.query_mode != "none":
        # The query LLM: the RAG model at temperature 0, so the same question is always rewritten the same way.
        store = QueryProcessingStore(store, create_llm(role="rag", temperature=0), args.query_mode)
    pipeline = RAGPipeline(persist_dir=persist_dir, llm=llm, reranker_config=ReRankerConfig(**overrides),
                           scorer=VectorOrderScorer() if no_reranker else None, vector_store=store)
    pipeline.cost_model = rag_model  # what question_cost() prices the answer tokens as

    # Index coverage: does ANY stored chunk contain each quote?
    stored = pipeline.vector_store.get(include=["documents", "metadatas"])
    all_chunks = stored["documents"]
    all_norms = [normalize(c) for c in all_chunks]
    if not all_chunks:
        raise SystemExit(f"The index at {persist_dir} is empty. Build it first.")
    index_info = {"persist_dir": str(Path(persist_dir).relative_to(ROOT)) if Path(persist_dir).is_relative_to(ROOT)
                  else persist_dir, "chunks": len(all_chunks),
                  "chunk_strategy": stored["metadatas"][0].get("chunk_strategy", "?")}

    # Warm-up: the first query pays one-off costs (loading the cross-encoder
    # weights, opening HTTP connections) that a running server pays only once.
    run_timed(pipeline, embedder, "warm-up question")
    index_tokens = sum(count_tokens(c) for c in all_chunks)
    index_cost = {"chunks": len(all_chunks), "tokens": index_tokens,
                  "usd": calculate_embedding_cost(index_tokens, embedding_model)}

    judge = None
    if not args.no_judge:
        judge = Judge()
        judge.calibrate()

    results = []
    print(f"Evaluating {len(cases)} cases against {len(all_chunks)} indexed chunks...\n")
    for case in cases:
        state, stages, cost = run_timed(pipeline, embedder, case["question"])
        answer = state.get("answer", "")
        r = {"id": case["id"], "type": case["type"], "question": case["question"],
             "answer": answer, "confidence": state.get("confidence"),
             "seconds": stages["total"], "latency": stages, "cost": cost,
             "contexts": context_record(state.get("context", []))}
        if args.query_mode != "none":
            r["search_queries"] = pipeline.vector_store.last_queries
        if judge:  # graded against exactly what the LLM saw, formatted as it saw it
            r["groundedness"] = judge.grade(case["question"], format_context(state.get("context", [])), answer)
            r["relevance"] = judge.relevance(case["question"], answer)

        if case["type"] == "unanswerable":
            r["refused"] = is_refusal(answer)
        else:
            candidates, context = state.get("candidates", []), state.get("context", [])
            in_index = [any(quote_in(ev["quote"], n) for n in all_norms) for ev in case["evidence"]]
            cand_hits = evidence_hits(case, candidates)
            ctx_hits = evidence_hits(case, context)
            first_rank = first_hit(cand_hits)  # best rank of ANY evidence
            # Re-scored after timing, so it doesn't inflate avg_seconds.
            rr_rank = first_hit(evidence_hits(case, rerank_order(pipeline, case["question"], candidates)))
            r.update({
                "evidence_in_index": sum(in_index) / len(in_index),
                "first_rank": None if first_rank is None else first_rank + 1,
                "reranked_rank": None if rr_rank is None else rr_rank + 1,
                "recall_any@1": first_rank is not None and first_rank < 1,
                "recall_any@5": first_rank is not None and first_rank < 5,
                "recall_any@top_k": first_rank is not None,
                "evidence_found@top_k": sum(bool(h) for h in cand_hits) / len(cand_hits),
                "context_recall_any": any(ctx_hits),
                "context_recall_all": all(ctx_hits),
                "correct": contains_expected(answer, case["must_contain_any"]),
            })
        results.append(r)
        mark = ("REFUSED" if r.get("refused") else "ANSWERED") if case["type"] == "unanswerable" \
            else ("✓" if r["correct"] else "✗")
        print(f"  {case['id']} {case['type']:12s} {mark:8s} rank={r.get('first_rank')!s:>4}"
              f"→{r.get('reranked_rank')!s:<4} "
              f"ctx={'yes' if r.get('context_recall_any') else '-':3s} [{r['confidence']}] "
              f"{grounded_mark(r):7s} {case['question'][:55]}")
        for claim in r.get("groundedness", {}).get("unsupported", []):
            print(f"        ↳ unsupported: {claim}")
        if r.get("relevance", {}).get("label") in ("partial", "does_not_answer"):
            print(f"        ↳ {r['relevance']['label']}: {r['relevance']['reason']}")

    summary = summarize(results)
    summary["top_k"] = pipeline.reranker_config.top_k_input
    summary["top_n"] = pipeline.reranker_config.top_n_output
    summary["cost"]["index_one_time"] = index_cost
    if judge:
        summary["groundedness"] = groundedness_summary(results, judge)
        summary["relevance"] = relevance_summary(results)
    print_report(summary)
    out = ROOT / "tests" / "evals" / "results" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The settings this run used, so compare_runs.py can show what changed between runs.
    cfg = load_config()
    if args.rag_temperature is not None:
        cfg["models"]["rag"]["temperature"] = args.rag_temperature
    cfg["models"]["embedding"]["name"] = embedding_model
    cfg["models"]["rag"]["name"] = rag_model
    cfg["rag"]["retrieval"]["top_k"] = pipeline.reranker_config.top_k_input
    cfg["rag"]["retrieval"]["top_n"] = pipeline.reranker_config.top_n_output
    cfg["models"]["reranker"]["name"] = "none" if no_reranker or not pipeline.rerank_enabled \
        else pipeline.reranker_config.model_name
    cfg["rag"]["query_processing"] = args.query_mode
    cfg["rag"]["reranker"]["min_score"] = None if no_reranker or not pipeline.rerank_enabled \
        else pipeline.reranker_config.min_score
    config = {"models": cfg.get("models"), "rag": cfg.get("rag"), "judge": not args.no_judge,
              "index": index_info}
    out.write_text(json.dumps({"label": args.label, "created": time.strftime("%Y-%m-%d %H:%M"),
                               "golden": args.golden, "config": config,
                               "summary": summary, "results": results}, indent=2))
    print(f"\nSaved → {out.relative_to(ROOT)}")


def summarize(results: list[dict]) -> dict:
    ans = [r for r in results if r["type"] != "unanswerable"]
    una = [r for r in results if r["type"] == "unanswerable"]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    ranks = [r["first_rank"] for r in ans]
    rr_ranks = [r["reranked_rank"] for r in ans]
    mrr = lambda ks: mean([1 / k if k else 0 for k in ks])  # noqa: E731
    at = lambda ks, n: mean([bool(k) and k <= n for k in ks])  # noqa: E731
    summary = {
        "answerable": len(ans), "unanswerable": len(una),
        "index_coverage": mean([r["evidence_in_index"] for r in ans]),
        "recall@1": mean([r["recall_any@1"] for r in ans]),
        "recall@5": mean([r["recall_any@5"] for r in ans]),
        "recall@top_k": mean([r["recall_any@top_k"] for r in ans]),
        "mrr@top_k": mrr(ranks),
        "reranker": {
            "mrr_before": mrr(ranks), "mrr_after": mrr(rr_ranks),
            "mrr_lift": round(mrr(rr_ranks) - mrr(ranks), 3),
            "recall@1_before": at(ranks, 1), "recall@1_after": at(rr_ranks, 1),
            "recall@5_before": at(ranks, 5), "recall@5_after": at(rr_ranks, 5),
            "improved": sum(bool(a) and (not b or a < b) for a, b in zip(rr_ranks, ranks)),
            "worsened": sum(bool(b) and (not a or a > b) for a, b in zip(rr_ranks, ranks)),
            "unchanged": sum(a == b for a, b in zip(rr_ranks, ranks)),
        },
        "context_recall_any": mean([r["context_recall_any"] for r in ans]),
        "context_recall_all": mean([r["context_recall_all"] for r in ans]),
        "answer_accuracy": mean([r["correct"] for r in ans]),
        "refusal_rate": mean([r["refused"] for r in una]),
        # Fallback or refusal. "unranked" (no re-ranker, so no gate) is not low confidence.
        "low_confidence_rate": mean([r["confidence"] in ("low", "none") for r in results]),
        "avg_seconds": mean([r["seconds"] for r in results]),
        "latency": latency_summary(results),
        "cost": cost_summary(results),
        "by_type": {},
    }
    groups = defaultdict(list)
    for r in ans:
        groups[r["type"]].append(r)
    for t, rs in groups.items():
        summary["by_type"][t] = {
            "n": len(rs),
            "recall@5": mean([r["recall_any@5"] for r in rs]),
            "context_recall": mean([r["context_recall_any"] for r in rs]),
            "accuracy": mean([r["correct"] for r in rs]),
        }
    return summary


def grounded_mark(r: dict) -> str:
    g = r.get("groundedness")
    if not g:
        return ""
    rel = {"answers": "R:✓", "partial": "R:½", "does_not_answer": "R:✗", "refusal": "R:-"}[r["relevance"]["label"]]
    return ("G:-" if not g["claims"] else f"G:{g['supported']}/{g['claims']}") + " " + rel


def relevance_summary(results: list[dict]) -> dict:
    """Relevance over answers that are not refusals; refusals are reported by question type,
    because refusing is right for an unanswerable question and a miss for an answerable one."""
    labels = [r["relevance"]["label"] for r in results]
    answered = [r for r in results if r["relevance"]["label"] != "refusal"]
    count = lambda lab: labels.count(lab)  # noqa: E731
    return {
        "answered": len(answered),
        "answers": count("answers"), "partial": count("partial"), "does_not_answer": count("does_not_answer"),
        "relevance_rate": round(count("answers") / len(answered), 3) if answered else None,
        "relevance_score": round(sum(RELEVANCE_SCORE[r["relevance"]["label"]] for r in answered) / len(answered), 3)
                           if answered else None,
        "refused_answerable": [r["id"] for r in results
                               if r["type"] != "unanswerable" and r["relevance"]["label"] == "refusal"],
        "refused_unanswerable": sum(r["type"] == "unanswerable" and r["relevance"]["label"] == "refusal" for r in results),
        "not_fully_relevant_ids": [r["id"] for r in answered if r["relevance"]["label"] != "answers"],
    }


def groundedness_summary(results: list[dict], judge: Judge) -> dict:
    """Over answers that make claims (refusals have none, so they can't be ungrounded)."""
    graded = [r for r in results if r["groundedness"]["claims"]]
    claims = sum(r["groundedness"]["claims"] for r in graded)
    supported = sum(r["groundedness"]["supported"] for r in graded)
    return {
        "judge_model": judge.model,
        "answers_with_claims": len(graded),
        "answers_without_claims": len(results) - len(graded),
        "grounded_rate": round(sum(r["groundedness"]["grounded"] for r in graded) / len(graded), 3) if graded else None,
        "claim_support_rate": round(supported / claims, 3) if claims else None,
        "claims": claims, "supported_claims": supported,
        "ungrounded_ids": [r["id"] for r in graded if not r["groundedness"]["grounded"]],
        "judge_usd_this_run": round(judge.usd, 4),
    }


def latency_summary(results: list[dict]) -> dict:
    """Seconds per stage over all questions: mean, p50, p95, and % of total time."""
    total = sum(r["latency"]["total"] for r in results) or 1
    out = {}
    for stage in STAGES:
        xs = [r["latency"][stage] for r in results]
        out[stage] = {"mean": round(sum(xs) / len(xs), 3), "p50": percentile(xs, 0.5),
                      "p95": percentile(xs, 0.95), "share": round(sum(xs) / total, 3)}
    return out


def cost_summary(results: list[dict]) -> dict:
    """Mean tokens and USD per question, the share each part costs, and per 1,000 questions."""
    n = len(results)
    avg = lambda k: sum(r["cost"][k] for r in results) / n  # noqa: E731
    total = avg("total_usd") or 1
    return {
        "models": {"embedding": results[0]["cost"]["embedding_model"], "llm": results[0]["cost"]["llm_model"]},
        "avg_embed_tokens": round(avg("embed_tokens"), 1),
        "avg_llm_input_tokens": round(avg("llm_input_tokens"), 1),
        "avg_llm_output_tokens": round(avg("llm_output_tokens"), 1),
        "avg_embed_usd": round(avg("embed_usd"), 8), "avg_llm_usd": round(avg("llm_usd"), 8),
        "avg_usd_per_question": round(avg("total_usd"), 8),
        "embed_share": round(avg("embed_usd") / total, 3), "llm_share": round(avg("llm_usd") / total, 3),
        "usd_per_1000_questions": round(avg("total_usd") * 1000, 4),
        "usd_this_run": round(avg("total_usd") * n, 6),
    }


def print_report(s: dict) -> None:
    print("\n" + "=" * 64)
    print(f"  Answerable: {s['answerable']}   Unanswerable: {s['unanswerable']}")
    print("-" * 64)
    print(f"  Index coverage (evidence stored at all) : {s['index_coverage']}")
    k = s.get("top_k", "k")
    print(f"  Vector recall@1 / @5 / @{k:<2}              : {s['recall@1']} / {s['recall@5']} / {s['recall@top_k']}")
    print(f"  MRR@{k:<2}                                  : {s['mrr@top_k']}")
    print(f"  (top_k={k}, top_n={s.get('top_n')} from config.yml → rag.retrieval)")
    rr = s["reranker"]
    print(f"  Re-ranker lift   before → after   (same {k} candidates, re-ordered)")
    print(f"    MRR                         : {rr['mrr_before']} → {rr['mrr_after']}  ({rr['mrr_lift']:+})")
    print(f"    recall@1                    : {rr['recall@1_before']} → {rr['recall@1_after']}")
    print(f"    recall@5                    : {rr['recall@5_before']} → {rr['recall@5_after']}")
    print(f"    questions improved/worse/same: {rr['improved']} / {rr['worsened']} / {rr['unchanged']}")
    print(f"  Context recall (what the LLM sees) any/all: {s['context_recall_any']} / {s['context_recall_all']}")
    print(f"  Answer accuracy (answerable)            : {s['answer_accuracy']}")
    print(f"  Refusal rate (unanswerable)             : {s['refusal_rate']}")
    print(f"  Low-confidence (gate fallback) rate     : {s['low_confidence_rate']}")
    if g := s.get("groundedness"):
        print(f"  Groundedness (judge: {g['judge_model']})")
        print(f"    answers fully grounded      : {g['grounded_rate']}  "
              f"(of {g['answers_with_claims']} answers with claims; {g['answers_without_claims']} refusals)")
        print(f"    claims supported            : {g['claim_support_rate']}  ({g['supported_claims']}/{g['claims']})")
        print(f"    ungrounded answers          : {', '.join(g['ungrounded_ids']) or 'none'}")
    if rel := s.get("relevance"):
        print(f"  Relevance (same judge, question + answer only)")
        print(f"    answers fully on target     : {rel['relevance_rate']}  "
              f"({rel['answers']} answers / {rel['partial']} partial / {rel['does_not_answer']} miss, "
              f"of {rel['answered']} non-refusals)")
        print(f"    relevance score (½ = partial): {rel['relevance_score']}")
        print(f"    not fully relevant          : {', '.join(rel['not_fully_relevant_ids']) or 'none'}")
        print(f"    refused answerable questions: {', '.join(rel['refused_answerable']) or 'none'}")
        print(f"    judge cost (eval only)      : ${g['judge_usd_this_run']}  (groundedness + relevance)")
    print(f"  Avg seconds / question                  : {s['avg_seconds']}")
    print("-" * 64)
    print(f"  {'latency (s)':12s} {'mean':>7s} {'p50':>7s} {'p95':>7s} {'share':>7s}")
    for stage, v in s["latency"].items():
        print(f"  {stage:12s} {v['mean']:>7.3f} {v['p50']:>7.3f} {v['p95']:>7.3f} {v['share']:>6.0%}")
    print("-" * 64)
    c = s["cost"]
    print(f"  cost / question   tokens (avg)                 USD (avg)   share")
    print(f"  embed  {c['models']['embedding']:22s} {c['avg_embed_tokens']:>8.1f}      ${c['avg_embed_usd']:.8f} {c['embed_share']:>6.1%}")
    print(f"  LLM    {c['models']['llm']:22s} {c['avg_llm_input_tokens']:>6.0f} in / {c['avg_llm_output_tokens']:.0f} out"
          f" ${c['avg_llm_usd']:.8f} {c['llm_share']:>6.1%}")
    print(f"  re-rank (local)                            0      $0")
    print(f"  total per question ${c['avg_usd_per_question']:.6f}  ·  per 1,000 questions ${c['usd_per_1000_questions']}")
    ix = c["index_one_time"]
    print(f"  one-time index: {ix['chunks']} chunks, {ix['tokens']:,} tokens → ${ix['usd']}")
    print("-" * 64)
    print(f"  {'type':12s} {'n':>3s} {'recall@5':>9s} {'ctx recall':>11s} {'accuracy':>9s}")
    for t, v in sorted(s["by_type"].items()):
        print(f"  {t:12s} {v['n']:>3d} {v['recall@5']!s:>9s} {v['context_recall']!s:>11s} {v['accuracy']!s:>9s}")
    print("=" * 64)


if __name__ == "__main__":
    main()
