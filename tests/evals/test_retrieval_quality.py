"""Retrieval-quality evaluation (marked `eval` — see tests/evals/README.md).

Unit tests prove the retrieval code RUNS. These measure whether it WORKS:
whether the chunk that answers the question is actually the chunk that
comes back, and whether the re-ranker earns the latency it costs.

Every threshold below is a regression floor set beneath current measured
performance — not a target. The printed number is the real output; the
assertion only catches collapse.
"""

from __future__ import annotations

from typing import List

import pytest

from tests.evals.conftest import retrieve_doc_ids

pytestmark = pytest.mark.eval


# --- Thresholds -------------------------------------------------------------
# recall@1 below this means the top hit is usually wrong, and since the
# re-ranker only reorders what retrieval returned, nothing downstream can
# recover from it.
MIN_RECALL_AT_1 = 0.66
# recall@3 is the number that matters for answer quality: top_n_output is 8,
# so a relevant chunk anywhere in the top few still reaches the prompt.
MIN_RECALL_AT_3 = 0.95
MIN_MRR = 0.75
# The gap between vector-only and reranked ordering. Zero lift means the
# cross-encoder is pure latency cost and should be reconsidered.
MIN_RERANKER_LIFT = 0.0
# Share of queries where the gate rejects every candidate and the pipeline
# falls back to raw vector order. Each one increments
# rag_retrieval_low_confidence_total in production. Currently ~0.17 on this
# corpus — see TestRerankerGate for why.
MAX_LOW_CONFIDENCE_RATE = 0.35


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top = retrieved[:k]
    return 1.0 if any(r in top for r in relevant) else 0.0


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    for position, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


class TestVectorRetrievalQuality:
    """Embedding search, before any re-ranking."""

    def test_recall_at_1(self, indexed_corpus, real_embedder, golden_set):
        cases = golden_set["retrieval_cases"]
        scores = [
            recall_at_k(retrieve_doc_ids(indexed_corpus, real_embedder, c["question"], 5),
                        c["relevant_doc_ids"], 1)
            for c in cases
        ]
        recall = sum(scores) / len(scores)
        print(f"\n  recall@1 = {recall:.3f}  ({int(sum(scores))}/{len(scores)} cases)")
        assert recall >= MIN_RECALL_AT_1, (
            f"recall@1 {recall:.3f} < {MIN_RECALL_AT_1}. The top-ranked chunk is "
            f"usually wrong — check the embedding model and chunk size before "
            f"blaming the re-ranker or the prompt."
        )

    def test_recall_at_3(self, indexed_corpus, real_embedder, golden_set):
        cases = golden_set["retrieval_cases"]
        scores = [
            recall_at_k(retrieve_doc_ids(indexed_corpus, real_embedder, c["question"], 5),
                        c["relevant_doc_ids"], 3)
            for c in cases
        ]
        recall = sum(scores) / len(scores)
        print(f"\n  recall@3 = {recall:.3f}")
        assert recall >= MIN_RECALL_AT_3, (
            f"recall@3 {recall:.3f} < {MIN_RECALL_AT_3}. If the answer is not in "
            f"the top 3 it will not reach the prompt — no downstream component "
            f"can fix this."
        )

    def test_mean_reciprocal_rank(self, indexed_corpus, real_embedder, golden_set):
        cases = golden_set["retrieval_cases"]
        ranks = [
            reciprocal_rank(retrieve_doc_ids(indexed_corpus, real_embedder, c["question"], 5),
                            c["relevant_doc_ids"])
            for c in cases
        ]
        mrr = sum(ranks) / len(ranks)
        print(f"\n  MRR = {mrr:.3f}")
        assert mrr >= MIN_MRR

    def test_per_case_breakdown(self, indexed_corpus, real_embedder, golden_set):
        """Diagnostic: prints which specific questions retrieve badly.

        The aggregate metrics say something regressed; this says what.
        """
        failures = []
        print()
        for case in golden_set["retrieval_cases"]:
            retrieved = retrieve_doc_ids(indexed_corpus, real_embedder, case["question"], 3)
            rank = reciprocal_rank(retrieved, case["relevant_doc_ids"])
            marker = "ok  " if rank == 1.0 else ("weak" if rank > 0 else "MISS")
            print(f"  [{marker}] rr={rank:.2f}  {case['question'][:55]!r} -> {retrieved[:3]}")
            if rank == 0.0:
                failures.append(case["question"])

        assert not failures, f"relevant chunk absent from top 3 for: {failures}"

    def test_distractor_does_not_outrank_the_answer(self, indexed_corpus, real_embedder):
        """The corpus contains a topically-similar but useless document
        (generic humidity theory). Semantic search is exactly what gets
        fooled by this — it is on-topic and answers nothing."""
        retrieved = retrieve_doc_ids(
            indexed_corpus, real_embedder,
            "What humidity accuracy does the Kestrel-7 report?", 3)
        assert retrieved[0] != "distractor-weather", (
            f"the generic-theory distractor outranked the spec sheet: {retrieved}"
        )


class TestRerankerLift:
    """Does the cross-encoder improve on vector order — on the real path?

    Important: production never calls ReRanker.re_rank() directly. It goes
    through select_context(policy="hybrid"), which falls back to raw vector
    order when the relevance gate rejects everything. Measuring re_rank()
    alone would report failures the application recovers from, and would
    miss that the recovery is happening at all.
    """

    def _retrieved_chunks(self, store, embedder, question, top_k=5):
        from src.business.rag.re_ranker.interface import RetrievedChunk

        result = store.query(embedder.embed_query(question), top_k)
        return [
            RetrievedChunk(chunk_id=i, text=d, metadata=m or {},
                           vector_score=float(s) if s is not None else 0.0)
            for i, d, m, s in zip(result["ids"][0], result["documents"][0],
                                  result["metadatas"][0], result["distances"][0])
        ]

    def _selected_ids(self, store, embedder, reranker, question):
        """The production path: rerank, then hybrid fallback."""
        from src.business.rag.re_ranker.orchestrator import select_context

        chunks = self._retrieved_chunks(store, embedder, question)
        selected, confidence = select_context(
            query=question, retrieved_chunks=chunks, reranker=reranker, policy="hybrid")
        return [c.chunk_id for c in selected], confidence

    def test_production_path_does_not_degrade_mrr(self, indexed_corpus, real_embedder,
                                                  real_reranker, golden_set):
        """MRR through select_context must not fall below vector-only.

        The hybrid fallback exists precisely so that a bad re-ranking cannot
        make the final answer worse than no re-ranking. If this fails, the
        fallback is not doing its job.
        """
        cases = golden_set["retrieval_cases"]

        vector_rr, selected_rr = [], []
        for case in cases:
            relevant = case["relevant_doc_ids"]
            vector_rr.append(reciprocal_rank(
                retrieve_doc_ids(indexed_corpus, real_embedder, case["question"], 5), relevant))
            ids, _ = self._selected_ids(indexed_corpus, real_embedder,
                                        real_reranker, case["question"])
            selected_rr.append(reciprocal_rank(ids, relevant))

        before = sum(vector_rr) / len(vector_rr)
        after = sum(selected_rr) / len(selected_rr)
        print(f"\n  MRR vector-only      = {before:.3f}")
        print(f"  MRR via select_context = {after:.3f}")
        print(f"  lift                   = {after - before:+.3f}")

        assert after - before >= MIN_RERANKER_LIFT, (
            f"the full retrieval path scores {before - after:.3f} WORSE than raw "
            f"vector order. The hybrid fallback should make this impossible — "
            f"check select_context() and the gate."
        )

    def test_confidence_is_reported_per_case(self, indexed_corpus, real_embedder,
                                             real_reranker, golden_set):
        """Diagnostic: how often does the gate reject everything?

        Each 'low' here is a query where the cross-encoder scored every
        candidate below min_score and the system fell back to vector order.
        In production each one increments rag_retrieval_low_confidence_total.
        A rising count is the early warning that retrieval is degrading.
        """
        print()
        low = 0
        for case in golden_set["retrieval_cases"]:
            ids, confidence = self._selected_ids(indexed_corpus, real_embedder,
                                                 real_reranker, case["question"])
            low += confidence == "low"
            hit = case["relevant_doc_ids"][0] in ids
            print(f"  [{confidence:4}] {'hit ' if hit else 'MISS'} {case['question'][:48]!r}")

        rate = low / len(golden_set["retrieval_cases"])
        print(f"\n  low-confidence fallback rate = {rate:.3f}")
        assert rate <= MAX_LOW_CONFIDENCE_RATE, (
            f"{rate:.0%} of queries fell back to raw vector order. The "
            f"re-ranker is rejecting almost everything — see the min_score "
            f"scale note in TestRerankerGate."
        )

    def test_gate_filters_the_irrelevant_distractor(self, indexed_corpus, real_embedder,
                                                    real_reranker):
        """min_score is the first hallucination firewall: it should keep
        weakly-related chunks out of the prompt entirely."""
        ids, confidence = self._selected_ids(
            indexed_corpus, real_embedder, real_reranker,
            "What was Veldrin's revenue in fiscal 2024?")
        print(f"\n  selected: {ids}  (confidence={confidence})")
        assert "veldrin-financials" in ids
        assert len(ids) < 5, "every candidate survived — the gate is not filtering"


class TestRerankerGate:
    """The min_score threshold and the scale it is applied to.

    FINDING pinned by these tests: CrossEncoderReRanker._batch_score()
    returns the model's raw logits — unbounded, measured roughly in
    [-10, +10] on this corpus — but ReRankerConfig.min_score defaults to
    0.15 and is documented as a relevance threshold, a value that only
    reads as sensible on a 0-1 probability scale.

    Applying 0.15 to a logit means the real gate is sigmoid(0.15) ~ 0.54,
    i.e. "at least 54% relevance probability" — roughly 3.5x stricter than
    the config's own comment implies. The hybrid fallback stops that
    causing wrong answers, but it converts precision gating into
    all-or-nothing: for an affected query the system silently reverts to
    unranked vector order.

    These tests document the current behaviour rather than asserting a fix.
    If the scoring is changed to emit sigmoid probabilities (or min_score is
    retuned for the logit scale), they should fail and be rewritten.
    """

    def _scores(self, store, embedder, reranker, question, top_k=5):
        from src.business.rag.re_ranker.interface import RetrievedChunk

        result = store.query(embedder.embed_query(question), top_k)
        chunks = [
            RetrievedChunk(chunk_id=i, text=d, metadata=m or {},
                           vector_score=float(s) if s is not None else 0.0)
            for i, d, m, s in zip(result["ids"][0], result["documents"][0],
                                  result["metadatas"][0], result["distances"][0])
        ]
        return reranker.scorer.score(question, chunks)

    def test_scores_are_logits_not_probabilities(self, indexed_corpus, real_embedder,
                                                 real_reranker):
        """If these ever land inside [0, 1], the scorer started applying a
        sigmoid and min_score=0.15 suddenly means something completely
        different. That is a silent, behaviour-changing event."""
        scored = self._scores(indexed_corpus, real_embedder, real_reranker,
                              "What temperature range does the Kestrel-7 operate in?")
        values = [c.rerank_score for c in scored]
        print(f"\n  rerank_score range: {min(values):.3f} .. {max(values):.3f}")
        assert min(values) < 0.0, (
            "all scores are non-negative — the scorer may now be emitting "
            "probabilities, which changes what min_score=0.15 gates on"
        )

    def test_a_relevant_chunk_can_be_gated_out_entirely(self, indexed_corpus,
                                                        real_embedder, real_reranker):
        """Pins the measured case.

        'What radio frequency does the sensor use in Europe?' retrieves the
        correct spec sheet at vector rank 1, but the cross-encoder scores it
        about -4.2 — below min_score — so the gate drops every candidate and
        re_rank() returns []. The answer IS in the corpus; the gate simply
        does not believe it.

        Production survives this via the hybrid fallback (asserted in
        TestRerankerLift). This test exists so that the underlying behaviour
        is visible rather than hidden behind that recovery.
        """
        question = "What radio frequency does the sensor use in Europe?"
        vector_ids = retrieve_doc_ids(indexed_corpus, real_embedder, question, 5)
        assert vector_ids[0] == "kestrel-specs", "vector search itself regressed"

        from src.business.rag.re_ranker.interface import RetrievedChunk
        result = indexed_corpus.query(real_embedder.embed_query(question), 5)
        chunks = [
            RetrievedChunk(chunk_id=i, text=d, metadata=m or {},
                           vector_score=float(s) if s is not None else 0.0)
            for i, d, m, s in zip(result["ids"][0], result["documents"][0],
                                  result["metadatas"][0], result["distances"][0])
        ]
        gated = real_reranker.re_rank(question, chunks)
        scored = {c.chunk_id: c.rerank_score for c in real_reranker.scorer.score(question, chunks)}

        print(f"\n  vector rank-1     : {vector_ids[0]}")
        print(f"  its rerank_score  : {scored['kestrel-specs']:.4f}")
        print(f"  min_score gate    : {real_reranker.config.min_score}")
        print(f"  survivors         : {[c.chunk_id for c in gated]}")

        assert gated == [], (
            "the gate no longer drops this case — if min_score or the scoring "
            "scale was fixed, delete this test and tighten "
            "MAX_LOW_CONFIDENCE_RATE"
        )

    def test_fallback_recovers_the_gated_chunk(self, indexed_corpus, real_embedder,
                                               real_reranker):
        """The other half of the pair: what the user actually gets."""
        from src.business.rag.re_ranker.orchestrator import select_context
        from src.business.rag.re_ranker.interface import RetrievedChunk

        question = "What radio frequency does the sensor use in Europe?"
        result = indexed_corpus.query(real_embedder.embed_query(question), 5)
        chunks = [
            RetrievedChunk(chunk_id=i, text=d, metadata=m or {},
                           vector_score=float(s) if s is not None else 0.0)
            for i, d, m, s in zip(result["ids"][0], result["documents"][0],
                                  result["metadatas"][0], result["distances"][0])
        ]
        selected, confidence = select_context(
            query=question, retrieved_chunks=chunks, reranker=real_reranker, policy="hybrid")

        print(f"\n  confidence = {confidence}")
        print(f"  selected   = {[c.chunk_id for c in selected]}")
        assert confidence == "low"
        assert "kestrel-specs" in [c.chunk_id for c in selected], (
            "the fallback failed to recover a chunk the gate dropped — this is "
            "a real answer-quality bug, not a threshold-tuning question"
        )


class TestEmbeddingConsistency:
    """Properties the whole index silently depends on."""

    def test_query_and_document_embeddings_share_dimensionality(self, real_embedder):
        """A mismatch makes every similarity search fail or return noise."""
        q = real_embedder.embed_query("test question")
        d = real_embedder.embed_documents(["test document"])[0]
        print(f"\n  dim = {len(q)}")
        assert len(q) == len(d)

    def test_identical_text_embeds_stably(self, real_embedder):
        """Repeated embeddings of the same text must be near-identical.

        NOT bit-identical: OpenAI's embedding endpoint returns values that
        differ in the last few floating-point digits between calls (measured
        drift ~6e-5 per component), so an equality assertion fails against a
        perfectly healthy API. What actually matters downstream is that the
        vectors stay in the same place — a query embedded twice must retrieve
        the same neighbours — so this asserts cosine similarity instead.
        """
        a = real_embedder.embed_query("the same sentence")
        b = real_embedder.embed_query("the same sentence")

        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        similarity = dot / norm

        print(f"\n  cos(same text, two calls) = {similarity:.9f}")
        assert similarity > 0.9999

    def test_related_text_scores_closer_than_unrelated(self, real_embedder):
        """The floor assumption under all of RAG. If this fails, nothing
        above it can work."""
        def cosine(u, v):
            dot = sum(a * b for a, b in zip(u, v))
            nu = sum(a * a for a in u) ** 0.5
            nv = sum(b * b for b in v) ** 0.5
            return dot / (nu * nv)

        anchor = real_embedder.embed_query("humidity sensor battery life")
        near = real_embedder.embed_query("how long does the sensor battery last")
        far = real_embedder.embed_query("medieval Portuguese poetry")

        near_score, far_score = cosine(anchor, near), cosine(anchor, far)
        print(f"\n  cos(related)   = {near_score:.3f}")
        print(f"  cos(unrelated) = {far_score:.3f}")
        assert near_score > far_score
