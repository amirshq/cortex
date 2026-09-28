# tests/business/rag/re_ranker/test_re_ranker.py

import sys
from pathlib import Path

# Add project root to Python path
# From: tests/business/rag/re_ranker/test_re_ranker.py
# Go up 5 levels: re_ranker -> rag -> business -> tests -> project_root
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.business.rag.re_ranker.re_ranker import ReRanker
from src.business.rag.re_ranker.config import ReRankerConfig
from src.business.rag.re_ranker.interface import RetrievedChunk
from tests.business.rag.re_ranker.mocks import MockScorer

"""
This function (_make_chunks) Manually constructs RetrievedChunk objects, Simulates what the VectorDB would return, 
Skips ingestion, chunking, and embedding entirely SO “Assume chunking has already happened correctly upstream.”
Because this is test code, so the function wants to check that “Given a list of retrieved chunks, does the re-ranker behave correctly?”
so it must just check the re-ranking process not the chunking logic, ingestion pipeline, embedding model, vector similarity,storage.
"""
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


def test_reranker_sorts_by_rerank_score():
    config = ReRankerConfig(top_n_output=3, min_score=0.0)
    reranker = ReRanker(scorer=MockScorer(), config=config)

    chunks = _make_chunks(5)
    result = reranker.re_rank("test query", chunks)

    scores = [c.rerank_score for c in result]

    assert scores == sorted(scores, reverse=True)
    assert len(result) == 3


def test_reranker_applies_gating():
    config = ReRankerConfig(min_score=0.3)
    reranker = ReRanker(scorer=MockScorer(), config=config)

    chunks = _make_chunks(5)
    result = reranker.re_rank("test query", chunks)

    assert all(c.rerank_score >= 0.3 for c in result)


def test_empty_input_returns_empty():
    config = ReRankerConfig()
    reranker = ReRanker(scorer=MockScorer(), config=config)

    assert reranker.re_rank("query", []) == []
