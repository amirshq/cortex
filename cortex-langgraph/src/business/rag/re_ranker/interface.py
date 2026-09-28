# Interface -> Contracts only
from dataclasses import dataclass
from typing import List, Dict, Any
from abc import ABC, abstractmethod


# =========================
# Data Contracts
# =========================

@dataclass
class RetrievedChunk:
    """
    Chunk returned from the VectorDB before re-ranking.
    """
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    vector_score: float


@dataclass
class ReRankedChunk(RetrievedChunk):
    """
    Chunk after re-ranking.
    Inherits all fields from RetrievedChunk and adds rerank_score.
    """
    rerank_score: float


# =========================
# Re-Ranker Interface
# =========================

class ReRankScorer(ABC):
    """
    Abstract base class for all re-ranking implementations.

    Responsibilities:
    - Score relevance between a query and retrieved chunks
    - Return chunks with rerank_score populated

    Non-responsibilities:
    - No gating (thresholding)
    - No sorting or truncation
    - No prompt construction
    - No VectorDB access
    """

    @abstractmethod
    def score(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[ReRankedChunk]:
        """
        Assign relevance scores to retrieved chunks.

        Args:
            query: User query string
            chunks: List of RetrievedChunk from VectorDB

        Returns:
            List of ReRankedChunk with rerank_score populated.
            Ordering is NOT enforced here.
        """
        raise NotImplementedError
