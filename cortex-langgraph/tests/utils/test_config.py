"""Tests for src/utils/config.py — the single source of truth for models."""

from __future__ import annotations

import pytest

import src.utils.config as config_module
from src.utils.config import load_config, model_settings

ROLES = ("chat", "rag", "embedding", "reranker")


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
