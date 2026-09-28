"""
This file is after all the re-ranker code and tests, so the final vector results or chunks which are re-ranked and 
filtered are passed to this orchestrator function to decide what to do next based on the selected policy and send
them to the LLM or not.

Option A — Fail Closed (strict)
    If re-ranker returns empty → do not call LLM
    Return a safe refusal (“I don’t have enough information…”)
    Best for high-risk domains
Option B — Fail Open (fallback)
    If empty → fall back to top-k vector results
    Proceed with caution
    Best for exploratory/internal tools
Option C — Hybrid (recommended default)
    Try strict re-ranking
    If empty → fall back to vector results and mark low confidence
    Log the event
"""
from typing import List, Tuple

from .re_ranker import ReRanker
from .interface import RetrievedChunk, ReRankedChunk


def select_context(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    reranker: ReRanker,
    policy: str = "hybrid",  # "fail_closed" | "fail_open" | "hybrid"
) -> Tuple[List[ReRankedChunk], str]:
    """
    Select final context for the LLM after re-ranking.

    Returns:
        chunks: final context chunks
        confidence: "high" | "low" | "none"
    """

    reranked_chunks = reranker.re_rank(query, retrieved_chunks)

    # Case 1: Re-ranker succeeded
    if reranked_chunks:
        return reranked_chunks, "high"

    # Case 2: Fail-closed
    if policy == "fail_closed":
        return [], "none"

    # Case 3: Fail-open / hybrid fallback
    fallback_chunks = retrieved_chunks[: reranker.config.top_n_output]

    return fallback_chunks, "low"