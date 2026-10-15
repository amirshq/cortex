"""Cross-encoder scorer — implements LangChain's `BaseCrossEncoder` interface.

A cross-encoder is a sequence-classification model that reads the query and a
document *together* and outputs one relevance score. It is slower than vector
similarity (one forward pass per pair) but much more accurate, which is why
it runs only on the top-k candidates the vector store already found.

cortex-core drove the model by hand: tokenizer + AutoModelForSequenceClassification
+ manual batching + torch.no_grad(). sentence-transformers' `CrossEncoder` does
all of that; this class adapts it to LangChain's `BaseCrossEncoder`
(`score(text_pairs) -> List[float]`) so it plugs into anything in the
LangChain ecosystem that accepts a cross-encoder.

Why not langchain_community's HuggingFaceCrossEncoder: langchain-community is
being retired. The interface lives in langchain_core, so implementing it here
is ten lines.

Score scale — kept identical to cortex-core on purpose: sentence-transformers
applies a sigmoid to single-label models by default, which would turn scores
into 0-1 probabilities. activation_fn=Identity keeps the raw logits cortex-core
produced, so ReRankerConfig.min_score gates exactly the same chunks in both
projects (see the TestRerankerGate finding in tests/evals for why that
threshold is on a logit scale).

They split re-ranking into two separate jobs: 
1- scoring (cross_encoder.py) and 
2- deciding what to keep (re_ranker.py).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from langchain_core.cross_encoders import BaseCrossEncoder

from .config import ReRankerConfig


class CrossEncoderScorer(BaseCrossEncoder):
    """Scores (query, document) pairs with a Hugging Face cross-encoder model."""

    def __init__(self, config: ReRankerConfig) -> None:
        from sentence_transformers import CrossEncoder

        self.config = config
        device = config.device if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(
            config.model_name,
            device=device,
            activation_fn=torch.nn.Identity(),
        )
        '''
        with this self.model = CrossEncoder(...)
        Now CrossEncoder internally manages things such as:
        model loading
        tokenizer
        tokenization
        batching
        forward pass
        inference
        conversion of outputs
        '''

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Relevance score for each (query, document) pair — raw logits."""
        if not text_pairs:
            return []
        scores = self.model.predict(
            text_pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]
