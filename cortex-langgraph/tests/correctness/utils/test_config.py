"""Tests for src/utils/config.py — the single source of truth for models."""

from __future__ import annotations

import pytest

import src.utils.config as config_module
from src.utils.config import load_config, model_settings

ROLES = ("chat", "rag", "embedding", "rag_embedding", "reranker", "judge")


class TestTheShippedConfig:
    def test_loads_without_duplicate_keys(self):
        """load_config() rejects duplicates, so this is the regression guard for
        the lost `llm_system_role` (two `llm_config:` blocks)."""
        assert load_config()

    @pytest.mark.parametrize("role", ROLES)
    def test_every_model_role_is_configured(self, role):
        assert model_settings(role)["name"]

    def test_rag_sampling_settings_are_configured(self):
        rag = model_settings("rag")
        assert isinstance(rag["temperature"], (int, float))
        assert isinstance(rag["max_tokens"], int)

    def test_rag_system_role_is_configured(self):
        assert load_config()["prompts"]["rag_system_role"].strip()

    def test_agent_tool_round_cap_is_configured(self):
        assert load_config()["agent"]["max_tool_rounds"] > 0


class TestDuplicateKeys:
    def test_duplicate_top_level_key_is_rejected(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text("llm_config:\n  a: 1\nother: 2\nllm_config:\n  b: 2\n")
        with pytest.raises(ValueError, match=r"Duplicate key 'llm_config'.*line 4"):
            load_config(path)

    def test_duplicate_nested_key_is_rejected(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text("models:\n  chat:\n    name: a\n    name: b\n")
        with pytest.raises(ValueError, match="Duplicate key 'name'"):
            load_config(path)

    def test_same_key_under_different_parents_is_fine(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text("models:\n  chat:\n    name: a\n  rag:\n    name: b\n")
        assert load_config(path)["models"]["rag"]["name"] == "b"


class TestModelSettings:
    def test_returns_a_copy(self):
        model_settings("rag")["name"] = "mutated"
        assert model_settings("rag")["name"] != "mutated"

    def test_missing_role_is_a_clear_error(self):
        with pytest.raises(KeyError, match="models.summary.name"):
            model_settings("summary")

    def test_role_without_a_name_is_a_clear_error(self, override_config):
        override_config({"models": {"chat": {"temperature": 0.1}}})
        with pytest.raises(KeyError, match="models.chat.name"):
            model_settings("chat")

    def test_reads_the_file_at_call_time(self, tmp_path, monkeypatch):
        path = tmp_path / "config.yml"
        path.write_text("models:\n  chat:\n    name: from-copy\n")
        monkeypatch.setattr(config_module, "CONFIG_PATH", path)
        assert model_settings("chat")["name"] == "from-copy"


class TestRagSettings:
    """RAG tunables live in config.yml → rag:, read by rag_settings(section)."""

    @pytest.mark.parametrize("section, keys", [
        ("chunking", {"chunk_size", "chunk_overlap"}),
        ("indexing", {"embedding_batch_size"}),
        ("retrieval", {"top_k", "top_n", "fallback_policy"}),
        ("reranker", {"min_score", "device", "batch_size", "blend_with_vector_score", "blend_alpha"}),
    ])
    def test_every_section_is_configured(self, section, keys):
        from src.utils.config import rag_settings

        assert keys <= set(rag_settings(section))

    def test_overlap_is_smaller_than_chunk_size(self):
        from src.utils.config import rag_settings

        c = rag_settings("chunking")
        assert c["chunk_overlap"] < c["chunk_size"]

    def test_top_n_does_not_exceed_top_k(self):
        from src.utils.config import rag_settings

        r = rag_settings("retrieval")
        assert r["top_n"] <= r["top_k"]

    def test_missing_section_is_a_clear_error(self, tmp_path, monkeypatch):
        from src.utils.config import rag_settings

        path = tmp_path / "config.yml"
        path.write_text("rag:\n  chunking:\n    chunk_size: 800\n    chunk_overlap: 100\n")
        monkeypatch.setattr(config_module, "CONFIG_PATH", path)
        with pytest.raises(KeyError, match="rag.retrieval"):
            rag_settings("retrieval")


class TestRagSettingsReachTheCode:
    """Changing only config.yml must change the pipeline — no code edit."""

    def test_reranker_config_reads_retrieval_and_gate_values(self, override_config):
        from src.business.rag.re_ranker.config import ReRankerConfig

        override_config({"rag": {
            "retrieval": {"top_k": 50, "top_n": 5, "fallback_policy": "hybrid"},
            "reranker": {"min_score": 1.5, "device": "cpu", "batch_size": 16,
                         "blend_with_vector_score": True, "blend_alpha": 0.3},
        }})
        c = ReRankerConfig()
        assert (c.top_k_input, c.top_n_output, c.min_score, c.batch_size) == (50, 5, 1.5, 16)
        assert (c.blend_with_vector_score, c.blend_alpha) == (True, 0.3)

    def test_explicit_arguments_still_override(self):
        from src.business.rag.re_ranker.config import ReRankerConfig

        assert ReRankerConfig(top_k_input=7).top_k_input == 7

    def test_chunking_reads_size_and_overlap(self, override_config):
        from langchain_core.documents import Document

        from src.business.rag.pdfingest.chunk import build_text_splitter, split_documents

        override_config({"rag": {"chunking": {"chunk_size": 300, "chunk_overlap": 30}}})
        splitter = build_text_splitter()
        assert (splitter._chunk_size, splitter._chunk_overlap) == (300, 30)
        chunks = split_documents([Document(page_content="word " * 400, metadata={"source_id": "a.pdf"})])
        assert chunks[0].metadata["chunk_strategy"] == "recursive_300_overlap_30"
        assert all(len(c.page_content) <= 300 for c in chunks)

    def test_fallback_policy_is_read_and_validated(self, override_config):
        from unittest.mock import MagicMock

        from src.business.rag.retrieval import RAGPipeline

        override_config({"rag": {"retrieval": {"top_k": 30, "top_n": 8, "fallback_policy": "fail_closed"}}})
        pipe = RAGPipeline(persist_dir="unused", vector_store=MagicMock(), scorer=MagicMock(), llm=MagicMock())
        assert pipe.policy == "fail_closed"

        override_config({"rag": {"retrieval": {"top_k": 30, "top_n": 8, "fallback_policy": "maybe"}}})
        with pytest.raises(ValueError, match="fallback_policy"):
            RAGPipeline(persist_dir="unused", vector_store=MagicMock(), scorer=MagicMock(), llm=MagicMock())
