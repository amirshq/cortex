"""Latency evaluation (marked `eval`).

Answers "where does the time actually go", per stage, so that a slow chat
endpoint can be attributed rather than guessed at. The budgets are
deliberately loose — they catch a stage getting an order of magnitude
slower (a model swap, a cold cross-encoder, an index that stopped using
its HNSW graph), not ordinary variance between runs.

Numbers are printed for every stage; read those rather than trusting the
pass/fail alone. Run with -s.
"""

from __future__ import annotations

import time
from typing import List

import pytest

from tests.evals.conftest import Stopwatch, percentile, retrieve_docs

pytestmark = pytest.mark.eval


# --- Budgets (seconds) ------------------------------------------------------
# One embedding call for a short query. Above this, suspect the network
# path or a switch to a larger embedding model.
EMBED_QUERY_P95_BUDGET = 2.0
# Vector search over a tiny local index. This should be milliseconds; a
# second means Chroma is doing a linear scan.
VECTOR_SEARCH_P95_BUDGET = 1.0
# Cross-encoder scoring of ~5 candidates on CPU. This is usually the
# largest non-LLM cost in the RAG path.
RERANK_P95_BUDGET = 10.0
# Full retrieve → rerank → generate.
END_TO_END_P95_BUDGET = 30.0
# Hermetic, no network: pure CPU work that should never be slow.
CHUNKING_BUDGET_PER_MB = 5.0


def report(label: str, samples: List[float]) -> None:
    print(f"\n  {label}")
    print(f"    n    = {len(samples)}")
    print(f"    min  = {min(samples):.3f}s")
    print(f"    p50  = {percentile(samples, 50):.3f}s")
    print(f"    p95  = {percentile(samples, 95):.3f}s")
    print(f"    max  = {max(samples):.3f}s")


class TestEmbeddingLatency:
    def test_query_embedding_p95(self, real_embedder, golden_set):
        samples = []
        for case in golden_set["retrieval_cases"]:
            with Stopwatch("embed") as sw:
                real_embedder.embed_query(case["question"])
            samples.append(sw.elapsed)

        report("embed_query", samples)
        assert percentile(samples, 95) < EMBED_QUERY_P95_BUDGET

    def test_batch_embedding_beats_serial_calls(self, real_embedder, golden_set):
        """index_builder batches by 50 for memory reasons; this confirms the
        batching is also a throughput win, not just an OOM guard."""
        texts = [d["text"] for d in golden_set["documents"]]

        with Stopwatch("batch") as batched:
            real_embedder.embed_documents(texts)
        with Stopwatch("serial") as serial:
            for text in texts:
                real_embedder.embed_documents([text])

        print(f"\n  batched({len(texts)}) = {batched.elapsed:.3f}s")
        print(f"  serial({len(texts)})  = {serial.elapsed:.3f}s")
        print(f"  speedup        = {serial.elapsed / max(batched.elapsed, 1e-9):.1f}x")
        assert batched.elapsed < serial.elapsed


class TestVectorSearchLatency:
    def test_search_p95(self, indexed_corpus, real_embedder, golden_set):
        """Timed with the embedding excluded, so this is search alone."""
        embeddings = [real_embedder.embed_query(c["question"])
                      for c in golden_set["retrieval_cases"]]

        samples = []
        for embedding in embeddings:
            with Stopwatch("search") as sw:
                indexed_corpus.similarity_search_by_vector_with_relevance_scores(embedding, k=5)
            samples.append(sw.elapsed)

        report("vector_search (embedding excluded)", samples)
        assert percentile(samples, 95) < VECTOR_SEARCH_P95_BUDGET

    def test_larger_top_k_does_not_blow_up(self, indexed_corpus, real_embedder):
        """Sub-linear scaling in k is the whole point of an ANN index."""
        embedding = real_embedder.embed_query("Kestrel-7 specifications")

        with Stopwatch("k=1") as small:
            indexed_corpus.similarity_search_by_vector_with_relevance_scores(embedding, k=1)
        with Stopwatch("k=30") as large:
            indexed_corpus.similarity_search_by_vector_with_relevance_scores(embedding, k=30)

        print(f"\n  k=1  = {small.elapsed:.4f}s")
        print(f"  k=30 = {large.elapsed:.4f}s")
        assert large.elapsed < small.elapsed * 10 + 0.5


class TestRerankerLatency:
    """Usually the dominant non-LLM cost — and the easiest to regress by
    accidentally moving it onto CPU or enlarging top_k_input."""

    def test_rerank_p95(self, indexed_corpus, real_embedder, real_reranker, golden_set):
        samples = []
        for case in golden_set["retrieval_cases"]:
            chunks = retrieve_docs(indexed_corpus, real_embedder, case["question"], 5)
            with Stopwatch("rerank") as sw:
                real_reranker.re_rank(case["question"], chunks)
            samples.append(sw.elapsed)

        report("rerank (5 candidates)", samples)
        assert percentile(samples, 95) < RERANK_P95_BUDGET

    def test_first_call_cold_start_is_reported(self, real_reranker):
        """The first scored query pays model warm-up. In a fresh container
        that cost lands on a real user's request — worth seeing explicitly
        rather than discovering in production p99s."""
        from langchain_core.documents import Document

        chunk = Document(id="c1", page_content="Some text about humidity sensors.", metadata={"vector_score": 0.1})
        with Stopwatch("cold") as cold:
            real_reranker.re_rank("humidity", [chunk])
        with Stopwatch("warm") as warm:
            real_reranker.re_rank("humidity", [chunk])

        print(f"\n  first call  = {cold.elapsed:.3f}s")
        print(f"  second call = {warm.elapsed:.3f}s")
        print(f"  warm-up cost ≈ {max(cold.elapsed - warm.elapsed, 0):.3f}s")


class TestEndToEndLatency:
    def test_full_rag_query_p95(self, rag_pipeline, golden_set):
        samples = []
        for case in golden_set["retrieval_cases"]:
            with Stopwatch("e2e") as sw:
                rag_pipeline.answer(case["question"])
            samples.append(sw.elapsed)

        report("full RAG answer() [retrieve + rerank + generate]", samples)
        assert percentile(samples, 95) < END_TO_END_P95_BUDGET

    def test_stage_attribution(self, rag_pipeline, golden_set):
        """Diagnostic, not a gate: splits one representative query into its
        stages so a regression can be attributed instead of guessed at.
        Calls the graph's own node functions in order, so each is timed alone."""
        question = golden_set["retrieval_cases"][0]["question"]
        state = {"question": question}

        with Stopwatch("retrieve") as retrieve:
            state.update(rag_pipeline._retrieve(state))
        with Stopwatch("rerank") as rerank:
            state.update(rag_pipeline._rerank(state))
            if not state["context"]:
                state.update(rag_pipeline._fallback(state))
        with Stopwatch("generate") as generate:
            rag_pipeline._generate(state)

        total = retrieve.elapsed + rerank.elapsed + generate.elapsed
        print(f"\n  question: {question!r}")
        for label, elapsed in (("retrieve", retrieve.elapsed),
                               ("rerank  ", rerank.elapsed),
                               ("generate", generate.elapsed)):
            print(f"    {label} = {elapsed:6.3f}s  ({elapsed / total * 100:5.1f}%)")
        print(f"    {'total   '} = {total:6.3f}s")


class TestHermeticLatency:
    """CPU-only work — no network, so these are stable enough to gate on."""

    def test_chunking_throughput(self):
        from langchain_core.documents import Document

        from src.business.rag.pdfingest.chunk import split_documents

        text = "word " * 200_000  # ~1 MB
        with Stopwatch("chunk") as sw:
            chunks = split_documents([Document(page_content=text, metadata={"source_id": "perf.pdf"})],
                                     chunk_size=800, overlap=100)

        mb = len(text) / 1_000_000
        print(f"\n  chunked {mb:.2f} MB into {len(chunks)} chunks in {sw.elapsed:.3f}s")
        print(f"  = {mb / max(sw.elapsed, 1e-9):.1f} MB/s")
        assert sw.elapsed / mb < CHUNKING_BUDGET_PER_MB

    def test_rate_limiter_overhead_is_negligible(self):
        """The limiter runs on the hot path of every single request."""
        from src.api.ratelimiter import TokenBucket

        bucket = TokenBucket(capacity=1_000_000, refill_rate=1_000_000)
        with Stopwatch("consume") as sw:
            for _ in range(100_000):
                bucket.consume(1)

        per_call_us = sw.elapsed / 100_000 * 1_000_000
        print(f"\n  100k consume() calls in {sw.elapsed:.3f}s = {per_call_us:.2f} µs/call")
        assert per_call_us < 100

    def test_prompt_building_is_not_a_bottleneck(self):
        from src.business.core.prompt_builder import build_agentic_system_prompt

        vector_results = [{"text": "recalled snippet " * 40} for _ in range(5)]
        with Stopwatch("prompt") as sw:
            for _ in range(1_000):
                build_agentic_system_prompt({"name": "A"}, vector_results, "summary")

        print(f"\n  1000 prompt builds in {sw.elapsed:.3f}s")
        assert sw.elapsed < 5.0
