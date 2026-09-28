# scripts/run_reranker_smoke_test.py

import sys
from pathlib import Path

# Add project root to Python path
# From: tests/business/rag/re_ranker/run_reranker_smoke_test.py
# Go up 4 levels: re_ranker -> rag -> business -> tests -> project_root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.business.rag.re_ranker.config import ReRankerConfig
from src.business.rag.re_ranker.cross_encoder import CrossEncoderReRanker
from src.business.rag.re_ranker.re_ranker import ReRanker
from src.business.rag.re_ranker.interface import RetrievedChunk


def main():
    # 1. Configuration
    config = ReRankerConfig(
        model_name="BAAI/bge-reranker-base",
        device="cpu",          # use cpu for first run
        top_k_input=5,
        top_n_output=3,
        min_score=-100.0       # disable gating for smoke test
    )

    # 2. Instantiate scorer
    scorer = CrossEncoderReRanker(config)

    # 3. Instantiate re-ranker
    reranker = ReRanker(
        scorer=scorer,
        config=config
    )

    # 4. Fake retrieved chunks (simulating VectorDB output)
    chunks = [
        RetrievedChunk(
            chunk_id="1",
            text="Paris is the capital of France.",
            metadata={"source": "wiki"},
            vector_score=0.91,
        ),
        RetrievedChunk(
            chunk_id="2",
            text="Berlin is the capital of Germany.",
            metadata={"source": "wiki"},
            vector_score=0.89,
        ),
        RetrievedChunk(
            chunk_id="3",
            text="The Eiffel Tower is located in Paris.",
            metadata={"source": "wiki"},
            vector_score=0.85,
        ),
    ]

    # 5. Query
    query = "What is the capital of France?"

    # 6. Run re-ranking
    results = reranker.re_rank(query, chunks)

    # 7. Print results
    print("\n=== RE-RANKER SMOKE TEST RESULTS ===\n")
    for i, chunk in enumerate(results, start=1):
        print(f"Rank {i}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Re-rank score: {chunk.rerank_score:.4f}")
        print(f"Text: {chunk.text}")
        print("-" * 40)


if __name__ == "__main__":
    main()
