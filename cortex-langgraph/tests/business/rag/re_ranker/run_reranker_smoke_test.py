# Manual smoke test: loads the REAL cross-encoder (downloads it on first run).
# Not collected by pytest (no test_ prefix). Run: python tests/business/rag/re_ranker/run_reranker_smoke_test.py

import sys
from pathlib import Path

# Add project root to Python path
# From: tests/business/rag/re_ranker/run_reranker_smoke_test.py
# Go up 4 levels: re_ranker -> rag -> business -> tests -> project_root
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document

from src.business.rag.re_ranker.config import ReRankerConfig
from src.business.rag.re_ranker.cross_encoder import CrossEncoderScorer
from src.business.rag.re_ranker.re_ranker import rerank_documents


def main():
    config = ReRankerConfig(
        model_name="BAAI/bge-reranker-base",
        device="cpu",          # use cpu for first run
        top_k_input=5,
        top_n_output=3,
        min_score=-100.0       # disable gating for smoke test
    )
    scorer = CrossEncoderScorer(config)

    query = "What is the capital of France?"
    docs = [
        Document(id="1", page_content="Paris is the capital and most populous city of France.", metadata={"vector_score": 0.2}),
        Document(id="2", page_content="Berlin is the capital of Germany.", metadata={"vector_score": 0.3}),
        Document(id="3", page_content="The Eiffel Tower is located in Paris.", metadata={"vector_score": 0.4}),
        Document(id="4", page_content="Bananas are a good source of potassium.", metadata={"vector_score": 0.5}),
    ]

    for doc in rerank_documents(query, docs, scorer, config):
        print(f"{doc.metadata['rerank_score']:8.3f}  {doc.page_content}")


if __name__ == "__main__":
    main()
