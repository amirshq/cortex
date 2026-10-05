"""Tests for the chat model factory (create_llm) and its helpers.

cortex-core tested its own BaseLLM / OpenAIModel / LocalHFModel classes.
Those are gone: every provider is now a LangChain BaseChatModel, so what's
left to test is OUR code — which class the factory picks, what it passes,
and how it fails. Constructing ChatOpenAI / AzureChatOpenAI makes no
network call, so these tests build the real classes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from src.business.core.model import azure_openai_settings, create_llm, model_label, resolve_provider
from src.utils.config import model_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from a known environment (the real .env may be loaded)."""
    for name in ("LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL_NAME", "HF_MODEL_NAME",
                 "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION",
                 "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def azure_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "chat-deployment")


class TestCreateLLMFactory:
    def test_default_provider_is_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        llm = create_llm()
        assert isinstance(llm, ChatOpenAI)
        assert isinstance(llm, BaseChatModel)

    def test_explicit_openai_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert isinstance(create_llm("openai"), ChatOpenAI)

    def test_env_var_selects_the_provider(self, monkeypatch, azure_env):
        monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
        assert isinstance(create_llm(), AzureChatOpenAI)

    def test_argument_overrides_env_var(self, monkeypatch, azure_env):
        monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert type(create_llm("openai")) is ChatOpenAI

    def test_case_insensitive_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert isinstance(create_llm("  OpenAI "), ChatOpenAI)

    def test_default_model_is_the_configured_chat_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert create_llm().model_name == model_settings("chat")["name"]

    def test_chat_role_model_comes_from_config(self, monkeypatch, override_config):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        override_config({"models": {"chat": {"name": "gpt-4.1"}}})
        assert create_llm(role="chat").model_name == "gpt-4.1"

    def test_rag_role_takes_model_and_sampling_from_config(self, monkeypatch, override_config):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        override_config({"models": {"rag": {"name": "gpt-4.1-mini", "temperature": 0.0, "max_tokens": 256}}})
        llm = create_llm(role="rag")
        assert (llm.model_name, llm.temperature, llm.max_tokens) == ("gpt-4.1-mini", 0.0, 256)

    def test_explicit_arguments_override_the_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        llm = create_llm(role="rag", model_name="gpt-4.1", temperature=0.2, max_tokens=64)
        assert (llm.model_name, llm.temperature, llm.max_tokens) == ("gpt-4.1", 0.2, 64)

    def test_openai_model_name_env_var_is_no_longer_read(self, monkeypatch):
        """Models live in config.yml only; a stray OPENAI_MODEL_NAME in an old
        .env must not silently win over it."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL_NAME", "some-other-model")
        assert create_llm().model_name == model_settings("chat")["name"]

    def test_unknown_role_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with pytest.raises(KeyError, match="models.summary.name"):
            create_llm(role="summary")

    def test_azure_uses_the_deployment_not_the_config_model_name(self, azure_env):
        """config.yml names OpenAI models; on Azure the deployment name is what
        must be sent. (cortex-core's RAG path sent "gpt-4o-mini" as the deployment.)"""
        assert create_llm("azure_openai", role="rag").deployment_name == "chat-deployment"

    def test_azure_still_gets_the_role_sampling_settings(self, azure_env):
        rag = model_settings("rag")
        llm = create_llm("azure_openai", role="rag")
        assert (llm.temperature, llm.max_tokens) == (rag["temperature"], rag["max_tokens"])

    def test_huggingface_requires_a_model_name(self):
        with pytest.raises(RuntimeError, match="HF_MODEL_NAME"):
            create_llm("huggingface")

    def test_custom_model_name(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert create_llm(model_name="gpt-4o-mini").model_name == "gpt-4o-mini"

    def test_passed_api_key_wins(self):
        llm = create_llm("openai", api_key="passed-key")
        assert llm.openai_api_key.get_secret_value() == "passed-key"

    def test_openai_requires_api_key(self):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            create_llm("openai")

    def test_sampling_settings_are_forwarded_when_given(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        llm = create_llm(temperature=0.7, max_tokens=512)
        assert llm.temperature == 0.7
        assert llm.max_tokens == 512

    def test_sampling_settings_are_left_to_the_provider_when_omitted(self, monkeypatch):
        """The agent never set them in cortex-core; the factory mustn't invent values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        llm = create_llm()
        assert llm.max_tokens is None
        assert llm.temperature is None

    def test_retries_are_enabled(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert create_llm().max_retries == 5

    def test_azure_openai_provider(self, azure_env):
        llm = create_llm("azure_openai")
        assert isinstance(llm, AzureChatOpenAI)
        assert llm.deployment_name == "chat-deployment"

    def test_azure_model_name_is_the_deployment(self, azure_env):
        assert create_llm("azure_openai", model_name="other-deployment").deployment_name == "other-deployment"

    def test_azure_openai_requires_deployment_name(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        with pytest.raises(RuntimeError, match="AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"):
            create_llm("azure_openai")

    def test_huggingface_provider_wraps_a_local_pipeline(self, monkeypatch):
        """Loading a real model needs a download + torch; the factory's job is
        only to wire ChatHuggingFace around a HuggingFacePipeline."""
        monkeypatch.setenv("HF_MODEL_NAME", "tiny/model")
        with patch("langchain_huggingface.HuggingFacePipeline.from_model_id") as from_model_id, \
             patch("langchain_huggingface.ChatHuggingFace") as chat_cls:
            llm = create_llm("huggingface", max_tokens=64)
        assert from_model_id.call_args.kwargs["model_id"] == "tiny/model"
        assert from_model_id.call_args.kwargs["task"] == "text-generation"
        assert from_model_id.call_args.kwargs["pipeline_kwargs"]["max_new_tokens"] == 64
        chat_cls.assert_called_once_with(llm=from_model_id.return_value)
        assert llm is chat_cls.return_value

    def test_unknown_provider_raises_error(self):
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            create_llm("llama-cpp")


class TestAzureOpenAISettings:
    def test_reads_settings_from_env(self, azure_env):
        assert azure_openai_settings() == ("https://example.invalid", "azure-key", "2024-10-21")

    def test_prefers_passed_api_key(self, azure_env):
        assert azure_openai_settings("passed-key")[1] == "passed-key"

    def test_api_version_env_var(self, azure_env, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
        assert azure_openai_settings()[2] == "2025-01-01"

    def test_missing_endpoint_raises_error(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
            azure_openai_settings()

    def test_missing_api_key_raises_error(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
        with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"):
            azure_openai_settings()


class TestResolveProvider:
    def test_defaults_to_openai(self):
        assert resolve_provider() == "openai"

    def test_normalises_case_and_whitespace(self):
        assert resolve_provider("  Azure_OpenAI ") == "azure_openai"

    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "huggingface")
        assert resolve_provider() == "huggingface"


class TestModelLabel:
    def test_openai_label_is_the_model_name(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert model_label(create_llm(model_name="gpt-4o-mini")) == "gpt-4o-mini"

    def test_azure_label_is_the_deployment(self, azure_env):
        assert model_label(create_llm("azure_openai")) == "chat-deployment"

    def test_falls_back_to_the_class_name(self):
        anonymous = MagicMock(spec=[])
        assert model_label(anonymous) == "MagicMock"
