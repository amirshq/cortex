"""Tests for the re-ranking policy (rerank_documents) and the cross-encoder adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document

from src.business.rag.re_ranker.config import ReRankerConfig
from src.business.rag.re_ranker.re_ranker import final_score, rerank_documents


class IndexScorer(BaseCrossEncoder):
    """Deterministic scorer: the i-th pair scores i * 0.1."""

    def score(self, text_pairs):
        return [i * 0.1 for i in range(len(text_pairs))]


def docs(n):
    return [Document(id=f"c{i}", page_content=f"text {i}", metadata={"vector_score": 1.0 - i * 0.05})
            for i in range(n)]


class TestRerankDocuments:
    def test_sorts_by_rerank_score(self):
        out = rerank_documents("q", docs(3), IndexScorer(), ReRankerConfig(min_score=0.0))
        assert [d.id for d in out] == ["c2", "c1", "c0"]

    def test_applies_gating(self):
        out = rerank_documents("q", docs(3), IndexScorer(), ReRankerConfig(min_score=0.15))
        assert [d.id for d in out] == ["c2"]

    def test_everything_gated_returns_empty(self):
        assert rerank_documents("q", docs(3), IndexScorer(), ReRankerConfig(min_score=9.0)) == []

    def test_empty_input_returns_empty(self):
        scorer = MagicMock(spec=BaseCrossEncoder)
        assert rerank_documents("q", [], scorer, ReRankerConfig()) == []
        scorer.score.assert_not_called()

    def test_limits_input_to_top_k_input(self):
        scorer = MagicMock(spec=BaseCrossEncoder)
        scorer.score.return_value = [1.0] * 4
        rerank_documents("q", docs(10), scorer, ReRankerConfig(top_k_input=4, min_score=0.0))
        assert len(scorer.score.call_args.args[0]) == 4

    def test_truncates_to_top_n_output(self):
        out = rerank_documents("q", docs(10), IndexScorer(), ReRankerConfig(top_n_output=2, min_score=0.0))
        assert len(out) == 2

    def test_scores_are_attached_without_mutating_the_input(self):
        original = docs(2)
        out = rerank_documents("q", original, IndexScorer(), ReRankerConfig(min_score=0.0))
        assert out[0].metadata["rerank_score"] == pytest.approx(0.1)
        assert out[0].metadata["vector_score"] == pytest.approx(0.95)
        assert all("rerank_score" not in d.metadata for d in original)

    def test_pairs_are_query_then_document_text(self):
        scorer = MagicMock(spec=BaseCrossEncoder)
        scorer.score.return_value = [1.0]
        rerank_documents("my question", docs(1), scorer, ReRankerConfig(min_score=0.0))
        assert scorer.score.call_args.args[0] == [("my question", "text 0")]


class TestReRankerConfigModel:
    def test_model_comes_from_config(self):
        from src.utils.config import model_settings

        assert ReRankerConfig().model_name == model_settings("reranker")["name"]

    def test_changing_the_config_changes_the_model(self, override_config):
        override_config({"models": {"reranker": {"name": "BAAI/bge-reranker-v2-m3"}}})
        assert ReRankerConfig().model_name == "BAAI/bge-reranker-v2-m3"

    def test_explicit_model_still_wins(self):
        assert ReRankerConfig(model_name="custom/model").model_name == "custom/model"


class TestFinalScore:
    def test_rerank_score_only_by_default(self):
        doc = Document(page_content="x", metadata={"rerank_score": 2.0, "vector_score": 0.5})
        assert final_score(doc, ReRankerConfig()) == 2.0

    def test_blends_with_vector_score_when_enabled(self):
        doc = Document(page_content="x", metadata={"rerank_score": 2.0, "vector_score": 0.5})
        config = ReRankerConfig(blend_with_vector_score=True, blend_alpha=0.5)
        assert final_score(doc, config) == pytest.approx(1.25)


class TestCrossEncoderScorer:
    """The adapter around sentence-transformers' CrossEncoder. The model itself
    is mocked — loading it needs a download (the evals load the real one)."""

    @pytest.fixture
    def cross_encoder_cls(self):
        with patch("sentence_transformers.CrossEncoder") as cls:
            cls.return_value.predict.return_value = np.array([2.5, -1.0])
            yield cls

    def test_is_a_langchain_cross_encoder(self, cross_encoder_cls):
        from src.business.rag.re_ranker.cross_encoder import CrossEncoderScorer

        assert isinstance(CrossEncoderScorer(ReRankerConfig()), BaseCrossEncoder)

    def test_loads_the_configured_model(self, cross_encoder_cls):
        from src.business.rag.re_ranker.cross_encoder import CrossEncoderScorer

        CrossEncoderScorer(ReRankerConfig(model_name="BAAI/bge-reranker-base"))
        assert cross_encoder_cls.call_args.args[0] == "BAAI/bge-reranker-base"

    def test_keeps_raw_logits_like_cortex_core(self, cross_encoder_cls):
        """sentence-transformers would sigmoid single-label scores into 0-1.
        Identity keeps the logit scale min_score was tuned against, so both
        projects gate exactly the same chunks."""
        from src.business.rag.re_ranker.cross_encoder import CrossEncoderScorer

        CrossEncoderScorer(ReRankerConfig())
        assert isinstance(cross_encoder_cls.call_args.kwargs["activation_fn"], torch.nn.Identity)

    def test_returns_plain_floats(self, cross_encoder_cls):
        from src.business.rag.re_ranker.cross_encoder import CrossEncoderScorer

        scores = CrossEncoderScorer(ReRankerConfig()).score([("q", "a"), ("q", "b")])
        assert scores == [2.5, -1.0]
        assert all(type(s) is float for s in scores)

    def test_uses_the_configured_batch_size(self, cross_encoder_cls):
        from src.business.rag.re_ranker.cross_encoder import CrossEncoderScorer

        CrossEncoderScorer(ReRankerConfig(batch_size=4)).score([("q", "a"), ("q", "b")])
        assert cross_encoder_cls.return_value.predict.call_args.kwargs["batch_size"] == 4

    def test_no_pairs_skips_the_model(self, cross_encoder_cls):
        from src.business.rag.re_ranker.cross_encoder import CrossEncoderScorer

        assert CrossEncoderScorer(ReRankerConfig()).score([]) == []
        cross_encoder_cls.return_value.predict.assert_not_called()
