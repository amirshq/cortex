"""
Here is the Business Logic of the Re-Ranker using Cross Encoder model.
VectorDB → RetrievedChunk[]
          ↓
CrossEncoderReRanker.score()
          ↓
ReRankedChunk[]
          ↓
ReRanker (gating, sorting, truncation)
          ↓
Prompt Builder

In a simple meaning, Cross Encoder is a sequence classification model 
that takes two texts as input and outputs a single score indicating their relevance.
"""

from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .interface import (
    RetrievedChunk,
    ReRankedChunk,
    ReRankScorer,
)
from .config import ReRankerConfig


class CrossEncoderReRanker(ReRankScorer):
    """
    Cross-encoder based re-ranker.

    Uses a transformer model to jointly encode (query, document)
    and output a relevance score.
    """

    def __init__(self, config: ReRankerConfig) -> None:
        self.config = config

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name
        ).to(self.device)

        self.model.eval()

    def score(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[ReRankedChunk]:
        """
        Compute relevance scores for query–chunk pairs.
        """

        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]

        scores = self._batch_score(pairs)

        return [
            ReRankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                vector_score=chunk.vector_score,
                rerank_score=float(score),
            )
            for chunk, score in zip(chunks, scores)
        ]

    # =========================
    # Internal helpers
    # =========================

    def _batch_score(self, pairs: List[tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs in batches.
        """

        all_scores: List[float] = []

        with torch.no_grad():
            for i in range(0, len(pairs), self.config.batch_size):
                batch = pairs[i : i + self.config.batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)

                logits = outputs.logits.squeeze(-1)

                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)

                all_scores.extend(logits.cpu().tolist())

        return all_scores
