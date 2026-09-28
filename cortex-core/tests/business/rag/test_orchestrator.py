import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
"""
At this point, the reranker has been tested to ensure it behaves correctly under scenatious like belows:
    Defined what happens when retrieval is weak
    Prevented accidental LLM hallucination
    Separated policy from ML
    Made behavior testable and explicit
"""
from src.business.rag.re_ranker.orchestrator import select_context
from src.business.rag.re_ranker.config import ReRankerConfig
from src.business.rag.re_ranker.re_ranker import ReRanker
from src.business.rag.re_ranker.interface import RetrievedChunk, ReRankedChunk, ReRankScorer


class EmptyScorer(ReRankScorer):
    """Always returns no reranked chunks."""
    def score(self, query, chunks):
        return []


def _make_chunks(n: int):
    return [
        RetrievedChunk(
            chunk_id=str(i),
            text=f"text {i}",
            metadata={},
            vector_score=1.0,
        )
        for i in range(n)
    ]


def test_fail_closed_policy():
    reranker = ReRanker(
        scorer=EmptyScorer(),
        config=ReRankerConfig(top_n_output=2)
    )

    chunks = _make_chunks(5)
    result, confidence = select_context(
        query="test",
        retrieved_chunks=chunks,
        reranker=reranker,
        policy="fail_closed"
    )

    assert result == []
    assert confidence == "none"


def test_fail_open_policy():
    reranker = ReRanker(
        scorer=EmptyScorer(),
        config=ReRankerConfig(top_n_output=2)
    )

    chunks = _make_chunks(5)
    result, confidence = select_context(
        query="test",
        retrieved_chunks=chunks,
        reranker=reranker,
        policy="fail_open"
    )

    assert len(result) == 2
    assert confidence == "low"