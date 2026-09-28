# tests/business/rag/re_ranker/mocks.py

from typing import List
from src.business.rag.re_ranker.interface import (
    RetrievedChunk,
    ReRankedChunk,
    ReRankScorer,
)


class MockScorer(ReRankScorer):
    """
    Deterministic scorer for unit tests.
    """

    def score(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[ReRankedChunk]:
        return [
            ReRankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                vector_score=chunk.vector_score,
                rerank_score=i * 0.1,  # deterministic score
            )
            for i, chunk in enumerate(chunks)
        ]
