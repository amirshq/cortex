"""Tests for the RAG vector store factory and reset.

cortex-core had to test its own Chroma/Azure adapters, including the
Azure→Chroma response-shape translation retrieval.py depended on. With
LangChain's VectorStore interface both backends return (Document, score)
pairs natively, so that translation layer (and its tests) are gone.

Chroma is exercised for real (in-process, tmp_path). Azure is mocked at the
class boundary — its constructor would call the service.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.business.rag.vector_store import (
    DEFAULT_AZURE_INDEX,
    create_vector_store,
    reset_vector_store,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("VECTOR_STORE_PROVIDER", "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY",
                 "AZURE_SEARCH_INDEX_NAME", "AZURE_SEARCH_EMBEDDING_DIM"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def azure_env(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "search-key")


class TestCreateVectorStoreFactory:
    def test_default_provider_is_chroma(self, tmp_path, fake_embedder):
        assert isinstance(create_vector_store(str(tmp_path), embedding=fake_embedder), Chroma)

    def test_env_var_selects_the_provider(self, tmp_path, fake_embedder, monkeypatch, azure_env):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "azure_search")
        with patch("langchain_azure_ai.vectorstores.AzureSearch") as azure:
            assert create_vector_store(str(tmp_path), embedding=fake_embedder) is azure.return_value

    def test_parameter_overrides_env_var(self, tmp_path, fake_embedder, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "azure_search")
        assert isinstance(create_vector_store(str(tmp_path), provider="chroma", embedding=fake_embedder), Chroma)

    def test_provider_name_is_case_insensitive(self, tmp_path, fake_embedder):
        assert isinstance(create_vector_store(str(tmp_path), provider=" Chroma ", embedding=fake_embedder), Chroma)

    def test_unknown_provider_raises(self, tmp_path, fake_embedder):
        with pytest.raises(ValueError, match="Unknown VECTOR_STORE_PROVIDER"):
            create_vector_store(str(tmp_path), provider="pinecone", embedding=fake_embedder)

    def test_default_embedding_comes_from_the_embedding_factory(self, tmp_path, fake_embedder):
        with patch("src.business.core.embedding.create_embedder", return_value=fake_embedder) as factory:
            store = create_vector_store(str(tmp_path))
        factory.assert_called_once_with()
        assert store.embeddings is fake_embedder

    def test_azure_requires_endpoint_and_key(self, tmp_path, fake_embedder):
        with pytest.raises(RuntimeError, match="AZURE_SEARCH_ENDPOINT"):
            create_vector_store(str(tmp_path), provider="azure_search", embedding=fake_embedder)

    def test_azure_uses_configured_index_and_dim(self, tmp_path, fake_embedder, monkeypatch, azure_env):
        monkeypatch.setenv("AZURE_SEARCH_INDEX_NAME", "my-index")
        monkeypatch.setenv("AZURE_SEARCH_EMBEDDING_DIM", "3072")
        with patch("langchain_azure_ai.vectorstores.AzureSearch") as azure:
            create_vector_store(str(tmp_path), provider="azure_search", embedding=fake_embedder)
        kwargs = azure.call_args.kwargs
        assert kwargs["azure_search_endpoint"] == "https://example.invalid"
        assert kwargs["azure_search_key"] == "search-key"
        assert kwargs["index_name"] == "my-index"
        assert kwargs["vector_search_dimensions"] == 3072
        assert kwargs["embedding_function"] is fake_embedder

    def test_azure_defaults(self, tmp_path, fake_embedder, azure_env):
        """Default index name differs from cortex-core's "rag-chunks" on purpose:
        LangChain's AzureSearch schema is not compatible with that index."""
        with patch("langchain_azure_ai.vectorstores.AzureSearch") as azure:
            create_vector_store(str(tmp_path), provider="azure_search", embedding=fake_embedder)
        assert azure.call_args.kwargs["index_name"] == DEFAULT_AZURE_INDEX != "rag-chunks"
        # Passing the dim up front stops AzureSearch from paying for a probe embedding.
        assert azure.call_args.kwargs["vector_search_dimensions"] == 1536


class TestChromaRoundTrip:
    """The contract retrieval.py relies on, exercised against real Chroma."""

    def test_add_then_search_returns_documents_with_distances(self, tmp_path, fake_embedder):
        store = create_vector_store(str(tmp_path), embedding=fake_embedder)
        store.add_documents([Document(page_content="alpha", metadata={"source_id": "a.pdf"})], ids=["c1"])

        (hit, distance), = store.similarity_search_with_score("alpha", k=1)
        assert hit.page_content == "alpha"
        assert hit.metadata == {"source_id": "a.pdf"}
        assert hit.id == "c1"
        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_re_adding_the_same_id_upserts(self, tmp_path, fake_embedder):
        """Stable chunk ids make re-indexing idempotent."""
        store = create_vector_store(str(tmp_path), embedding=fake_embedder)
        store.add_documents([Document(page_content="v1")], ids=["c1"])
        store.add_documents([Document(page_content="v2")], ids=["c1"])
        assert store.get()["documents"] == ["v2"]

    def test_persists_across_instances(self, tmp_path, fake_embedder):
        create_vector_store(str(tmp_path), embedding=fake_embedder).add_documents(
            [Document(page_content="kept")], ids=["c1"])
        assert create_vector_store(str(tmp_path), embedding=fake_embedder).get()["documents"] == ["kept"]


class TestResetVectorStore:
    def test_chroma_reset_empties_the_collection(self, tmp_path, fake_embedder):
        store = create_vector_store(str(tmp_path), embedding=fake_embedder)
        store.add_documents([Document(page_content="stale")], ids=["c1"])

        reset_vector_store(str(tmp_path))

        assert create_vector_store(str(tmp_path), embedding=fake_embedder).get()["ids"] == []

    def test_chroma_reset_on_an_empty_directory_is_fine(self, tmp_path):
        reset_vector_store(str(tmp_path / "never-used"))

    def test_chroma_reset_leaves_other_collections_alone(self, tmp_path, fake_embedder):
        other = create_vector_store(str(tmp_path), collection_name="other", embedding=fake_embedder)
        other.add_documents([Document(page_content="keep me")], ids=["k1"])
        reset_vector_store(str(tmp_path))
        assert other.get()["documents"] == ["keep me"]

    def test_azure_reset_drops_the_index(self, azure_env):
        with patch("azure.search.documents.indexes.SearchIndexClient") as client_cls:
            reset_vector_store("unused", provider="azure_search")
        client_cls.return_value.delete_index.assert_called_once_with(DEFAULT_AZURE_INDEX)

    def test_azure_reset_tolerates_a_missing_index(self, azure_env):
        from azure.core.exceptions import ResourceNotFoundError

        with patch("azure.search.documents.indexes.SearchIndexClient") as client_cls:
            client_cls.return_value.delete_index.side_effect = ResourceNotFoundError("gone")
            reset_vector_store("unused", provider="azure_search")

    def test_reset_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown VECTOR_STORE_PROVIDER"):
            reset_vector_store("unused", provider="pinecone")
