"""Tests for the embedding factory (create_embedder) and the embedding cache.

cortex-core tested its own Embedder ABC, OpenAIEmbedder and a hand-written
retry loop. Those are replaced by LangChain's Embeddings interface and
OpenAIEmbeddings / AzureOpenAIEmbeddings (which retry internally). What's
left to test is the factory's selection logic and the CacheBackedEmbeddings
wiring. Constructing the real classes makes no network call.
"""

from __future__ import annotations

from typing import List

import pytest
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from src.business.core.embedding import create_embedder, with_embedding_cache
from src.utils.config import model_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("EMBEDDING_PROVIDER", "OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                 "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def azure_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "embed-deployment")


class CountingEmbedder(Embeddings):
    """Counts how many texts actually reach the (would-be paid) model."""

    def __init__(self):
        self.embedded: List[str] = []

    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        self.embedded.append(text)
        return [float(len(text)), 1.0]


class TestCreateEmbedderFactory:
    def test_default_provider_is_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        embedder = create_embedder()
        assert isinstance(embedder, OpenAIEmbeddings)
        assert isinstance(embedder, Embeddings)

    def test_default_model_comes_from_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert create_embedder().model == model_settings("embedding")["name"]

    def test_changing_the_config_changes_the_model(self, monkeypatch, override_config):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        override_config({"models": {"embedding": {"name": "text-embedding-3-large"}}})
        assert create_embedder().model == "text-embedding-3-large"

    def test_azure_uses_the_deployment_not_the_config_model_name(self, azure_env):
        assert create_embedder("azure_openai").deployment == "embed-deployment"

    def test_custom_model_name(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert create_embedder(model="text-embedding-3-large").model == "text-embedding-3-large"

    def test_passed_api_key_wins(self):
        embedder = create_embedder("openai", api_key="passed-key")
        assert embedder.openai_api_key.get_secret_value() == "passed-key"

    def test_openai_requires_api_key(self):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            create_embedder("openai")

    def test_retries_are_enabled(self, monkeypatch):
        """Replaces cortex-core's hand-rolled exponential backoff."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert create_embedder().max_retries == 5

    def test_azure_openai_provider(self, azure_env):
        embedder = create_embedder("azure_openai")
        assert isinstance(embedder, AzureOpenAIEmbeddings)
        assert embedder.deployment == "embed-deployment"

    def test_azure_requires_embedding_deployment_name(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        with pytest.raises(RuntimeError, match="AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"):
            create_embedder("azure_openai")

    def test_env_var_provider_selection(self, monkeypatch, azure_env):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "azure_openai")
        assert isinstance(create_embedder(), AzureOpenAIEmbeddings)

    def test_case_insensitive_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert isinstance(create_embedder(" OPENAI "), OpenAIEmbeddings)

    def test_unknown_provider_raises_error(self):
        with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
            create_embedder("cohere")

    def test_no_cache_by_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert type(create_embedder()) is OpenAIEmbeddings

    def test_cache_dir_wraps_the_model_in_a_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        embedder = create_embedder(cache_dir=str(tmp_path))
        assert type(embedder).__name__ == "CacheBackedEmbeddings"
        assert isinstance(embedder.underlying_embeddings, OpenAIEmbeddings)


class TestEmbeddingCache:
    def test_repeated_documents_are_embedded_once(self, tmp_path):
        inner = CountingEmbedder()
        cached = with_embedding_cache(inner, cache_dir=str(tmp_path), namespace="test:model")

        first = cached.embed_documents(["alpha", "beta"])
        second = cached.embed_documents(["alpha", "beta", "gamma"])

        assert inner.embedded == ["alpha", "beta", "gamma"]
        assert second[:2] == first

    def test_repeated_queries_are_embedded_once(self, tmp_path):
        inner = CountingEmbedder()
        cached = with_embedding_cache(inner, cache_dir=str(tmp_path), namespace="test:model")
        cached.embed_query("what is x?")
        cached.embed_query("what is x?")
        assert inner.embedded == ["what is x?"]

    def test_cache_survives_a_new_instance(self, tmp_path):
        """It's an on-disk cache: a restart must not re-pay for embeddings."""
        with_embedding_cache(CountingEmbedder(), cache_dir=str(tmp_path), namespace="m").embed_documents(["alpha"])
        inner = CountingEmbedder()
        with_embedding_cache(inner, cache_dir=str(tmp_path), namespace="m").embed_documents(["alpha"])
        assert inner.embedded == []

    def test_different_models_never_share_cache_entries(self, tmp_path):
        """Vectors from different models are not interchangeable."""
        with_embedding_cache(CountingEmbedder(), cache_dir=str(tmp_path), namespace="openai:small").embed_documents(["alpha"])
        inner = CountingEmbedder()
        with_embedding_cache(inner, cache_dir=str(tmp_path), namespace="openai:large").embed_documents(["alpha"])
        assert inner.embedded == ["alpha"]

    def test_namespace_with_separator_characters_is_usable(self, tmp_path):
        """Regression: LocalFileStore rejects ':' in keys, and the factory's
        namespace is "<provider>:<model>". The first write used to raise."""
        cached = with_embedding_cache(CountingEmbedder(), cache_dir=str(tmp_path),
                                      namespace="azure_openai:my deployment/v2")
        assert cached.embed_documents(["alpha"]) == [[5.0, 1.0]]

    def test_factory_built_cache_can_write(self, monkeypatch, tmp_path):
        """End-to-end through create_embedder(cache_dir=...): the real
        namespace it builds must be accepted by the store."""
        from unittest.mock import patch

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("langchain_openai.OpenAIEmbeddings.embed_documents", return_value=[[0.1, 0.2]]) as inner:
            embedder = create_embedder(cache_dir=str(tmp_path))
            assert embedder.embed_documents(["alpha"]) == [[0.1, 0.2]]
            assert embedder.embed_documents(["alpha"]) == [[0.1, 0.2]]
        inner.assert_called_once()
