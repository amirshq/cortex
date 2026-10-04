"""Step 4 — Re-ranking: keep only the chunks that really answer the question.

The embedding model compares the question and a chunk SEPARATELY (two vectors).
A re-ranker (a "cross-encoder") reads the question and the chunk TOGETHER and
scores how well the chunk answers the question. It is much more accurate, but
too slow to run over a whole database — so we run it only on the top-k results
from retrieval, and keep the best few.

The model runs locally (downloaded once from Hugging Face, ~1 GB), no API key.
"""

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model):
        self.model = CrossEncoder(model)

    def rerank(self, question, chunks, top_n):
        """Score every chunk against the question; return the best top_n, best first."""
        if not chunks:
            return []
        scores = self.model.predict([(question, chunk["text"]) for chunk in chunks])
        for chunk, score in zip(chunks, scores):
            chunk["score"] = float(score)  # 0 = irrelevant ... 1 = highly relevant
        return sorted(chunks, key=lambda chunk: chunk["score"], reverse=True)[:top_n]
