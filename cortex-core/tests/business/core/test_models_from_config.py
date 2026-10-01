"""Every model comes from config.yml → models: — changing the config alone
changes the model, and no factory falls back to a model named in code."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.config import load_config, model_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("LLM_PROVIDER", "OPENAI_MODEL_NAME", "HF_MODEL_NAME",
                 "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class TestCreateLLM:
    def test_rag_role_takes_model_and_sampling_from_config(self):
        from src.business.core.model import create_llm

        rag = model_settings("rag")
        llm = create_llm(role="rag")
        assert (llm.model_name, llm.temperature, llm.max_tokens) == \
            (rag["name"], rag["temperature"], rag["max_tokens"])

    def test_changing_the_config_changes_the_model(self, override_config):
        from src.business.core.model import create_llm

        override_config({"models": {"rag": {"name": "gpt-4.1-mini", "temperature": 0.0, "max_tokens": 256}}})
        llm = create_llm(role="rag")
        assert (llm.model_name, llm.temperature, llm.max_tokens) == ("gpt-4.1-mini", 0.0, 256)

    def test_openai_model_name_env_var_is_no_longer_read(self, monkeypatch):
        from src.business.core.model import create_llm

        monkeypatch.setenv("OPENAI_MODEL_NAME", "some-other-model")
        assert create_llm().model_name == model_settings("rag")["name"]

    def test_azure_rag_uses_the_chat_deployment(self, monkeypatch):
        """This path used to send the hardcoded "gpt-4o-mini" as the deployment."""
        from src.business.rag.retrieval import RAGPipeline

        monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
        monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "my-chat-deployment")
        with patch("src.business.rag.retrieval.create_embedder"), \
             patch("src.business.rag.retrieval.create_vector_store"), \
             patch("src.business.rag.retrieval.CrossEncoderReRanker"), \
             patch("src.business.core.model.build_azure_openai_client"):
            pipeline = RAGPipeline(persist_dir="unused")
        assert pipeline.llm.model_name == "my-chat-deployment"

    def test_huggingface_requires_a_model_name(self):
        from src.business.core.model import create_llm

        with pytest.raises(RuntimeError, match="HF_MODEL_NAME"):
            create_llm("huggingface")


class TestOtherModels:
    def test_embedding_model_comes_from_config(self, override_config):
        from src.business.core.embedding import create_embedder

        assert create_embedder().model == model_settings("embedding")["name"]
        override_config({"models": {"embedding": {"name": "text-embedding-3-large"}}})
        assert create_embedder().model == "text-embedding-3-large"

    def test_reranker_model_comes_from_config(self, override_config):
        from src.business.rag.re_ranker.config import ReRankerConfig

        assert ReRankerConfig().model_name == model_settings("reranker")["name"]
        override_config({"models": {"reranker": {"name": "BAAI/bge-reranker-v2-m3"}}})
        assert ReRankerConfig().model_name == "BAAI/bge-reranker-v2-m3"

    def test_agent_model_comes_from_config(self, override_config):
        from src.business.chatbot.agentic_chatbot import AgenticChatbot

        override_config({"models": {"chat": {"name": "gpt-4.1"}}})
        with patch("src.business.chatbot.agentic_chatbot.load_dotenv"):
            bot = AgenticChatbot(long_term_memory=MagicMock(), redis_memory=MagicMock(),
                                 user_id="u", live_data_provider=MagicMock())
        assert bot.model_name == "gpt-4.1"


class TestRagSystemRole:
    def test_default_system_role_is_the_configured_one(self):
        """Regression: a duplicate `llm_config:` key in config.yml used to wipe
        the configured role, so the RAG prompt silently fell back to the
        generic "You are a helpful assistant."."""
        from src.business.core.prompt_builder import PromptBuilder

        configured = load_config()["prompts"]["rag_system_role"]
        assert PromptBuilder().system_prompt == configured
        assert configured != "You are a helpful assistant."


class TestReportedModel:
    @pytest.mark.asyncio
    async def test_chat_response_reports_the_model_that_ran(self):
        """model_used was hardcoded to "gpt-4o", so metrics mislabelled any
        other configured model."""
        from src.business.chatbot import process_chat_message
        from src.database.dto import ChatMessageRequest

        bot = MagicMock(model_name="gpt-4.1")

        async def chat(**kwargs):
            return "hi"

        bot.chat = chat
        with patch("src.business.chatbot._make_chatbot", return_value=bot):
            result = await process_chat_message(ChatMessageRequest(message="hello", user_id=1))
        assert result["model_used"] == "gpt-4.1"
