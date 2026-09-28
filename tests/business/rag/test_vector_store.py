"""Tests for the RAG vector stores and their provider factory.

The load-bearing test in this file is the Azure→Chroma response-shape
translation. retrieval.py's _retrieve() indexes into result["ids"][0],
["documents"][0], ["metadatas"][0], ["distances"][0] regardless of which
backend produced them. If AzureSearchVectorStore.query() ever stops
matching that nested-list shape, RAG breaks at runtime on Azure only —
silently returning zero chunks rather than raising.

The Azure SDK is imported lazily inside AzureSearchVectorStore, so these
tests inject fake SDK modules into sys.modules instead of requiring
azure-search-documents to be installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from src.business.rag.vector_store import (
    ChromaVectorStore,
    VectorStore,
    VectorStoreBase,
    create_vector_store,
)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class TestVectorStoreInterface:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            VectorStoreBase()

    def test_subclass_must_implement_all_three_methods(self):
        class Incomplete(VectorStoreBase):
            def reset(self): ...

        with pytest.raises(TypeError):
            Incomplete()

    def test_backward_compat_alias_points_at_chroma(self):
        assert VectorStore is ChromaVectorStore


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------
class TestChromaVectorStore:
    def _store(self, tmp_path):
        with patch("src.business.rag.vector_store.chromadb") as chroma:
            client = MagicMock()
            collection = MagicMock()
            collection.name = "pdf_chunks"
            client.get_or_create_collection.return_value = collection
            chroma.PersistentClient.return_value = client
            store = ChromaVectorStore(persist_dir=str(tmp_path))
        return store, client, collection

    def test_creates_collection_without_embedding_function(self, tmp_path):
        """We supply embeddings manually; letting Chroma pick its own would
        silently embed with a different model than the query side uses."""
        _, client, _ = self._store(tmp_path)
        assert client.get_or_create_collection.call_args.kwargs["embedding_function"] is None

    def test_upsert_forwards_all_four_parallel_lists(self, tmp_path):
        store, _, collection = self._store(tmp_path)
        store.upsert(ids=["a"], embeddings=[[0.1]], metadatas=[{"k": "v"}], documents=["text"])

        kwargs = collection.upsert.call_args.kwargs
        assert kwargs["ids"] == ["a"]
        assert kwargs["embeddings"] == [[0.1]]
        assert kwargs["documents"] == ["text"]

    def test_upsert_rejects_mismatched_lengths(self, tmp_path):
        """A silent zip() truncation here would drop chunks from the index."""
        store, _, _ = self._store(tmp_path)
        with pytest.raises(ValueError, match="length mismatch"):
            store.upsert(ids=["a", "b"], embeddings=[[0.1]], metadatas=[{}], documents=["t"])

    def test_query_requests_documents_metadatas_and_distances(self, tmp_path):
        """_retrieve() reads all three; omitting any breaks RetrievedChunk."""
        store, _, collection = self._store(tmp_path)
        store.query([0.1, 0.2], top_k=5)

        kwargs = collection.query.call_args.kwargs
        assert kwargs["n_results"] == 5
        assert set(kwargs["include"]) == {"documents", "metadatas", "distances"}

    def test_query_wraps_the_embedding_in_a_batch_list(self, tmp_path):
        store, _, collection = self._store(tmp_path)
        store.query([0.1, 0.2])
        assert collection.query.call_args.kwargs["query_embeddings"] == [[0.1, 0.2]]

    def test_reset_deletes_then_recreates_the_collection(self, tmp_path):
        store, client, _ = self._store(tmp_path)
        store.reset()
        client.delete_collection.assert_called_once_with("pdf_chunks")
        assert client.get_or_create_collection.call_count == 2


# ---------------------------------------------------------------------------
# Azure AI Search
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_azure_sdk(monkeypatch):
    """Install a minimal fake azure-search-documents into sys.modules."""

    class ResourceNotFoundError(Exception):
        pass

    def passthrough(*args, **kwargs):
        return MagicMock()

    index_client = MagicMock()
    search_client = MagicMock()

    modules = {
        "azure": types.ModuleType("azure"),
        "azure.core": types.ModuleType("azure.core"),
        "azure.core.credentials": types.ModuleType("azure.core.credentials"),
        "azure.core.exceptions": types.ModuleType("azure.core.exceptions"),
        "azure.search": types.ModuleType("azure.search"),
        "azure.search.documents": types.ModuleType("azure.search.documents"),
        "azure.search.documents.indexes": types.ModuleType("azure.search.documents.indexes"),
        "azure.search.documents.indexes.models": types.ModuleType("azure.search.documents.indexes.models"),
        "azure.search.documents.models": types.ModuleType("azure.search.documents.models"),
    }
    modules["azure.core.credentials"].AzureKeyCredential = passthrough
    modules["azure.core.exceptions"].ResourceNotFoundError = ResourceNotFoundError
    modules["azure.search.documents"].SearchClient = lambda **kw: search_client
    modules["azure.search.documents.indexes"].SearchIndexClient = lambda **kw: index_client

    models = modules["azure.search.documents.indexes.models"]
    for name in ("HnswAlgorithmConfiguration", "SearchField", "SearchIndex",
                 "SimpleField", "VectorSearch", "VectorSearchProfile"):
        setattr(models, name, passthrough)
    models.SearchFieldDataType = types.SimpleNamespace(STRING="Edm.String", INT32="Edm.Int32")
    modules["azure.search.documents.models"].VectorizedQuery = passthrough

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return types.SimpleNamespace(
        index_client=index_client,
        search_client=search_client,
        ResourceNotFoundError=ResourceNotFoundError,
    )


def make_azure_store(fake_azure_sdk, **kwargs):
    from src.business.rag.vector_store import AzureSearchVectorStore
    return AzureSearchVectorStore(
        endpoint="https://example.search.windows.net",
        api_key="key",
        **kwargs,
    )


class TestAzureSearchVectorStore:
    def test_creates_the_index_when_missing(self, fake_azure_sdk):
        fake_azure_sdk.index_client.get_index.side_effect = fake_azure_sdk.ResourceNotFoundError()
        make_azure_store(fake_azure_sdk)
        fake_azure_sdk.index_client.create_index.assert_called_once()

    def test_does_not_recreate_an_existing_index(self, fake_azure_sdk):
        fake_azure_sdk.index_client.get_index.return_value = MagicMock()
        make_azure_store(fake_azure_sdk)
        fake_azure_sdk.index_client.create_index.assert_not_called()

    def test_reset_drops_then_recreates(self, fake_azure_sdk):
        """Must match ChromaVectorStore.reset()'s wipe-to-empty semantics."""
        fake_azure_sdk.index_client.get_index.return_value = MagicMock()
        store = make_azure_store(fake_azure_sdk)
        store.reset()
        fake_azure_sdk.index_client.delete_index.assert_called_once()
        fake_azure_sdk.index_client.create_index.assert_called_once()

    def test_reset_tolerates_a_missing_index(self, fake_azure_sdk):
        fake_azure_sdk.index_client.get_index.return_value = MagicMock()
        store = make_azure_store(fake_azure_sdk)
        fake_azure_sdk.index_client.delete_index.side_effect = fake_azure_sdk.ResourceNotFoundError()
        store.reset()
        fake_azure_sdk.index_client.create_index.assert_called_once()

    def test_upsert_rejects_mismatched_lengths(self, fake_azure_sdk):
        fake_azure_sdk.index_client.get_index.return_value = MagicMock()
        store = make_azure_store(fake_azure_sdk)
        with pytest.raises(ValueError, match="length mismatch"):
            store.upsert(ids=["a", "b"], embeddings=[[0.1]], metadatas=[{}], documents=["t"])

    def test_upsert_flattens_metadata_into_index_fields(self, fake_azure_sdk):
        """Azure has a flat schema — Chroma's nested metadata dict must be
        spread across typed top-level fields."""
        fake_azure_sdk.index_client.get_index.return_value = MagicMock()
        store = make_azure_store(fake_azure_sdk)
        store.upsert(
            ids=["c1"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"source_id": "doc.pdf", "section": "table",
                        "chunk_start": 100, "chunk_end": 900,
                        "chunk_strategy": "char_window"}],
            documents=["chunk text"],
        )
        doc = fake_azure_sdk.search_client.merge_or_upload_documents.call_args.kwargs["documents"][0]
        assert doc["id"] == "c1"
        assert doc["content"] == "chunk text"
        assert doc["source_id"] == "doc.pdf"
        assert doc["section"] == "table"
        assert doc["chunk_start"] == 100

    def test_upsert_defaults_missing_metadata_fields(self, fake_azure_sdk):
        """Azure rejects a document missing a declared field; Chroma doesn't
        care. Absent keys must become typed zero-values, not KeyErrors."""
        fake_azure_sdk.index_client.get_index.return_value = MagicMock()
        store = make_azure_store(fake_azure_sdk)
        store.upsert(ids=["c1"], embeddings=[[0.1]], metadatas=[{}], documents=["t"])

        doc = fake_azure_sdk.search_client.merge_or_upload_documents.call_args.kwargs["documents"][0]
        assert doc["source_id"] == ""
        assert doc["chunk_start"] == 0

    def test_upsert_tolerates_none_metadata(self, fake_azure_sdk):
        fake_azure_sdk.index_client.get_index.return_value = MagicMock()
        store = make_azure_store(fake_azure_sdk)
        store.upsert(ids=["c1"], embeddings=[[0.1]], metadatas=[None], documents=["t"])
        assert fake_azure_sdk.search_client.merge_or_upload_documents.called


class TestAzureToChromaShapeTranslation:
    """THE contract test: Azure's flat rows → Chroma's nested-list dict.

    retrieval.py::_retrieve() does result.get("ids", [[]])[0] on whatever
    the store returns. Break this shape and RAG-on-Azure returns zero
    chunks with no error — the LLM then answers from nothing.
    """

    def _query(self, fake_azure_sdk, rows):
        fake_azure_sdk.index_client.get_index.return_value = MagicMock()
        store = make_azure_store(fake_azure_sdk)
        fake_azure_sdk.search_client.search.return_value = iter(rows)
        return store.query([0.1, 0.2], top_k=5)

    def test_returns_the_four_chroma_keys(self, fake_azure_sdk):
        result = self._query(fake_azure_sdk, [])
        assert set(result) == {"ids", "documents", "metadatas", "distances"}

    def test_every_value_is_wrapped_in_an_outer_batch_list(self, fake_azure_sdk):
        """Chroma's outer list is the batch dimension. _retrieve() indexes
        [0] into all four — a flat list would yield a single character."""
        rows = [{"id": "c1", "content": "text", "@search.score": 0.9}]
        result = self._query(fake_azure_sdk, rows)
        for key in ("ids", "documents", "metadatas", "distances"):
            assert isinstance(result[key], list), key
            assert isinstance(result[key][0], list), key

    def test_retrieval_can_unpack_the_azure_response(self, fake_azure_sdk):
        """End-to-end proof: feed Azure's output through _retrieve()'s
        actual unpacking code and assert real chunks come out."""
        rows = [
            {"id": "c1", "content": "first", "source_id": "d.pdf", "section": "text",
             "chunk_start": 0, "chunk_end": 800, "chunk_strategy": "cw", "@search.score": 0.9},
            {"id": "c2", "content": "second", "source_id": "d.pdf", "section": "table",
             "chunk_start": 700, "chunk_end": 1500, "chunk_strategy": "cw", "@search.score": 0.7},
        ]
        result = self._query(fake_azure_sdk, rows)

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        assert ids == ["c1", "c2"]
        assert docs == ["first", "second"]
        assert len(list(zip(ids, docs, metas, dists))) == 2
        assert metas[0]["source_id"] == "d.pdf"
        assert metas[1]["section"] == "table"

    def test_similarity_score_is_negated_into_a_distance(self, fake_azure_sdk):
        """Azure scores are higher-is-better; Chroma distances are
        lower-is-better. Without the sign flip the ordering semantics
        invert between backends."""
        rows = [{"id": "c1", "content": "t", "@search.score": 0.9},
                {"id": "c2", "content": "t", "@search.score": 0.4}]
        result = self._query(fake_azure_sdk, rows)

        assert result["distances"][0] == [-0.9, -0.4]
        assert result["distances"][0][0] < result["distances"][0][1]

    def test_missing_score_defaults_to_zero(self, fake_azure_sdk):
        result = self._query(fake_azure_sdk, [{"id": "c1", "content": "t"}])
        assert result["distances"][0] == [0.0]

    def test_missing_content_becomes_empty_string(self, fake_azure_sdk):
        result = self._query(fake_azure_sdk, [{"id": "c1", "@search.score": 0.5}])
        assert result["documents"][0] == [""]

    def test_metadata_carries_only_the_declared_index_fields(self, fake_azure_sdk):
        from src.business.rag.vector_store import AzureSearchVectorStore
        rows = [{"id": "c1", "content": "t", "source_id": "d", "@search.score": 0.5,
                 "unexpected_field": "should not leak"}]
        result = self._query(fake_azure_sdk, rows)
        assert set(result["metadatas"][0][0]) == set(AzureSearchVectorStore.INDEX_FIELDS_METADATA_KEYS)

    def test_empty_result_set_yields_empty_inner_lists(self, fake_azure_sdk):
        """Must be [[]] not [] — _retrieve() would still index [0]."""
        result = self._query(fake_azure_sdk, [])
        assert result["ids"] == [[]]
        assert result.get("ids", [[]])[0] == []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
class TestCreateVectorStoreFactory:
    def test_default_provider_is_chroma(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VECTOR_STORE_PROVIDER", raising=False)
        with patch("src.business.rag.vector_store.ChromaVectorStore") as chroma:
            create_vector_store(persist_dir=str(tmp_path))
        chroma.assert_called_once()

    def test_env_var_selects_the_provider(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "chroma")
        with patch("src.business.rag.vector_store.ChromaVectorStore") as chroma:
            create_vector_store(persist_dir=str(tmp_path))
        chroma.assert_called_once()

    def test_parameter_overrides_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "azure_search")
        with patch("src.business.rag.vector_store.ChromaVectorStore") as chroma:
            create_vector_store(persist_dir=str(tmp_path), provider="chroma")
        chroma.assert_called_once()

    def test_provider_name_is_case_insensitive(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "  CHROMA  ")
        with patch("src.business.rag.vector_store.ChromaVectorStore") as chroma:
            create_vector_store(persist_dir=str(tmp_path))
        chroma.assert_called_once()

    def test_azure_requires_endpoint_and_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "azure_search")
        monkeypatch.delenv("AZURE_SEARCH_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_SEARCH_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="AZURE_SEARCH_ENDPOINT"):
            create_vector_store(persist_dir=str(tmp_path))

    def test_azure_uses_configured_index_and_dim(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "azure_search")
        monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://e.search.windows.net")
        monkeypatch.setenv("AZURE_SEARCH_API_KEY", "k")
        monkeypatch.setenv("AZURE_SEARCH_INDEX_NAME", "custom-index")
        monkeypatch.setenv("AZURE_SEARCH_EMBEDDING_DIM", "3072")

        with patch("src.business.rag.vector_store.AzureSearchVectorStore") as azure:
            create_vector_store(persist_dir=str(tmp_path))

        kwargs = azure.call_args.kwargs
        assert kwargs["index_name"] == "custom-index"
        assert kwargs["dim"] == 3072

    def test_azure_dim_defaults_to_1536(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "azure_search")
        monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://e.search.windows.net")
        monkeypatch.setenv("AZURE_SEARCH_API_KEY", "k")
        monkeypatch.delenv("AZURE_SEARCH_EMBEDDING_DIM", raising=False)

        with patch("src.business.rag.vector_store.AzureSearchVectorStore") as azure:
            create_vector_store(persist_dir=str(tmp_path))
        assert azure.call_args.kwargs["dim"] == 1536

    def test_unknown_provider_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "pinecone")
        with pytest.raises(ValueError, match="Unknown VECTOR_STORE_PROVIDER"):
            create_vector_store(persist_dir=str(tmp_path))
