"""Tests for create_conversation_vector_store — the long-term-memory index factory.

cortex-core tested a hand-written ChromaVectorDB adapter (add/search/delete
forwarding into chromadb). That adapter is replaced by langchain_chroma.Chroma,
so what's left is the factory's selection logic and the one behaviour we
depend on it for: persistence.
"""

from __future__ import annotations

import pytest
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

from src.memory.vectordb import create_conversation_vector_store


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("CHAT_VECTOR_STORE_PROVIDER", raising=False)
    monkeypatch.delenv("VECTOR_STORE_PROVIDER", raising=False)


def build(tmp_path, embedder, **kwargs):
    return create_conversation_vector_store(
        collection_name="chat_history",
        embedding=embedder,
        persist_directory=str(tmp_path / "store"),
        **kwargs,
    )


class TestCreateConversationVectorStoreFactory:
    def test_default_provider_is_chroma(self, tmp_path, fake_embedder):
        store = build(tmp_path, fake_embedder)
        assert isinstance(store, Chroma)
        assert isinstance(store, VectorStore)

    def test_env_var_selects_the_provider(self, tmp_path, fake_embedder, monkeypatch):
        monkeypatch.setenv("CHAT_VECTOR_STORE_PROVIDER", "azure_search")
        with pytest.raises(NotImplementedError):
            build(tmp_path, fake_embedder)

    def test_parameter_overrides_env_var(self, tmp_path, fake_embedder, monkeypatch):
        monkeypatch.setenv("CHAT_VECTOR_STORE_PROVIDER", "azure_search")
        assert isinstance(build(tmp_path, fake_embedder, provider="chroma"), Chroma)

    def test_provider_name_is_case_insensitive(self, tmp_path, fake_embedder):
        assert isinstance(build(tmp_path, fake_embedder, provider=" CHROMA "), Chroma)

    def test_azure_search_is_not_implemented_yet(self, tmp_path, fake_embedder):
        with pytest.raises(NotImplementedError, match="not implemented"):
            build(tmp_path, fake_embedder, provider="azure_search")

    def test_unknown_provider_raises(self, tmp_path, fake_embedder):
        with pytest.raises(ValueError, match="Unknown CHAT_VECTOR_STORE_PROVIDER"):
            build(tmp_path, fake_embedder, provider="pinecone")

    def test_this_switch_is_independent_of_the_rag_one(self, tmp_path, fake_embedder, monkeypatch):
        """RAG chunks and conversation memory are separate indexes by design."""
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "azure_search")
        assert isinstance(build(tmp_path, fake_embedder), Chroma)

    def test_uses_the_given_embedding_model(self, tmp_path, fake_embedder):
        assert build(tmp_path, fake_embedder).embeddings is fake_embedder


class TestPersistence:
    def test_memories_survive_a_new_client(self, tmp_path, fake_embedder):
        """cortex-core's chromadb.Client(Settings(persist_directory=...)) was an
        in-memory client in chromadb 1.x, so long-term memory was silently lost
        on every restart. This store must actually persist."""
        build(tmp_path, fake_embedder).add_texts(["remember me"], metadatas=[{"user_id": "u1"}], ids=["m1"])

        reopened = build(tmp_path, fake_embedder)
        assert reopened.get(ids=["m1"])["documents"] == ["remember me"]

    def test_writes_to_the_persist_directory(self, tmp_path, fake_embedder):
        build(tmp_path, fake_embedder).add_texts(["x"], ids=["m1"])
        assert any((tmp_path / "store").iterdir())
