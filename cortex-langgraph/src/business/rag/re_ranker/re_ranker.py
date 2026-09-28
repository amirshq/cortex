# Some part of this project is for the RAG system and some parts are for the chatbot system.
# The Re-Ranker is only used in the RAG system.

from typing import List

from .config import ReRankerConfig
from .interface import (
    RetrievedChunk,
    ReRankedChunk,
    ReRankScorer,
)


class ReRanker:
    """
    Re-ranking orchestrator.

    Responsibilities:
    - Call the scoring model
    - Apply gating logic
    - Sort chunks by relevance
    - Truncate to top-n
    - Return prompt-ready chunks

    This class contains NO ML logic.
    """

    def __init__(
        self,
        scorer: ReRankScorer,
        config: ReRankerConfig,
    ) -> None:
        self.scorer = scorer
        self.config = config

    def re_rank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[ReRankedChunk]:
        """
        Execute the re-ranking pipeline.

        Args:
            query: User query
            chunks: Retrieved chunks from VectorDB

        Returns:
            List of re-ranked, filtered, sorted chunks
        """

        if not chunks:
            return []

        # 1. Limit input size (recall control)
        candidates = chunks[: self.config.top_k_input]

        # 2. Score relevance (model-dependent)
        scored_chunks = self.scorer.score(
            query=query,
            chunks=candidates,
        )

        # 3. Apply gating (hallucination control)
        gated_chunks = self._apply_gating(scored_chunks)

        if not gated_chunks:
            return []

        # 4. Sort by relevance score (descending)
        ranked_chunks = sorted(
            gated_chunks,
            key=self._final_score,
            reverse=True,
        )

        # 5. Truncate to top-n (context control)
        return ranked_chunks[: self.config.top_n_output]

    # =========================
    # Internal helpers
    # =========================

    def _apply_gating(
        self,
        chunks: List[ReRankedChunk],
    ) -> List[ReRankedChunk]:
        """
        Drop chunks below minimum relevance threshold.
        """
        return [
            chunk
            for chunk in chunks
            if chunk.rerank_score >= self.config.min_score
        ]

    def _final_score(
        self,
        chunk: ReRankedChunk,
    ) -> float:
        """
        Compute final ranking score.
        Supports optional blending with vector score.
        """
        if not self.config.blend_with_vector_score:
            return chunk.rerank_score

        alpha = self.config.blend_alpha
        return (
            alpha * chunk.rerank_score
            + (1 - alpha) * chunk.vector_score
        )
