"""Re-ranking logic: score → gate → sort → truncate.

This is the part of cortex-core's re-ranker that has no LangChain equivalent.
LangChain's `CrossEncoderReranker` (a document compressor) keeps the top-n by
score but throws the scores away and has no relevance threshold — and the
threshold is the RAG path's first hallucination firewall. So the scoring model
is a LangChain `BaseCrossEncoder`, and the policy around it stays ours.

What moved out: cortex-core's `orchestrator.select_context()` (what to do when
the gate rejects everything). That decision is now a conditional edge in the
RAG graph (retrieval.py), where it's visible as a branch.

Chunks are LangChain `Document`s; scores ride along in their metadata:
  metadata["vector_score"]  — distance from the vector store (lower = closer)
  metadata["rerank_score"]  — cross-encoder relevance (higher = more relevant)
"""

from __future__ import annotations

from typing import List

from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document

from .config import ReRankerConfig


def rerank_documents(
    query: str,
    documents: List[Document],
    scorer: BaseCrossEncoder,
    config: ReRankerConfig,
) -> List[Document]:
    """
    Execute the re-ranking pipeline.

    Returns re-ranked, gated, sorted, truncated chunks — possibly an empty
    list when nothing clears the relevance gate.
    """
    if not documents:
        return []

    # 1. Limit input size (recall control)
    candidates = documents[: config.top_k_input]

    # 2. Score relevance (model-dependent)
    scores = scorer.score([(query, doc.page_content) for doc in candidates])
    scored = [
        Document(
            id=doc.id,
            page_content=doc.page_content,
            metadata={**doc.metadata, "rerank_score": float(score)},
        )
        for doc, score in zip(candidates, scores)
    ]

    # 3. Apply gating (hallucination control)
    gated = [doc for doc in scored if doc.metadata["rerank_score"] >= config.min_score]
    if not gated:
        return []

    # 4. Sort by relevance score (descending)
    ranked = sorted(gated, key=lambda doc: final_score(doc, config), reverse=True)

    # 5. Truncate to top-n (context control)
    return ranked[: config.top_n_output]


def final_score(doc: Document, config: ReRankerConfig) -> float:
    """
    Compute final ranking score.
    Supports optional blending with vector score.
    """
    rerank_score = doc.metadata["rerank_score"]
    if not config.blend_with_vector_score:
        return rerank_score

    alpha = config.blend_alpha
    return alpha * rerank_score + (1 - alpha) * doc.metadata.get("vector_score", 0.0)
