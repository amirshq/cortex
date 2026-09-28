"""Tests for ChromaVectorDB — the conversation-memory vector store.

Deliberately a SEPARATE index from the RAG store (different metadata
shape), so it has its own factory and its own switch. The response
translation here differs from the RAG store's too: Chroma's nested lists
are flattened into a list of dicts, because LongTermMemory.recall()'s
callers iterate results and read r["text"].
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.memory.vectordb import (
    ChromaVectorDB,
    ConversationVectorStoreBase,
    VectorDB,
    create_conversation_vector_store,
)


@pytest.fixture
def collection():
    return MagicMock()


@pytest.fixture
def db(collection):
    with patch("src.memory.vectordb.Client") as client_cls:
        client = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_cls.return_value = client
        return ChromaVectorDB(collection_name="chat_history", persist_directory="/tmp/x")


class TestInterface:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            ConversationVectorStoreBase()

    def test_subclass_must_implement_all_three_methods(self):
        class Incomplete(ConversationVectorStoreBase):
            def add(self, ids, embeddings, documents, metadatas): ...

        with pytest.raises(TypeError):
            Incomplete()

    def test_backward_compat_alias(self):
        assert VectorDB is ChromaVectorDB

    def test_chroma_db_satisfies_the_interface(self, db):
        assert isinstance(db, ConversationVectorStoreBase)


class TestAdd:
    def test_forwards_all_four_lists(self, db, collection):
        db.add(ids=["i1"], embeddings=[[0.1]], documents=["text"], metadatas=[{"user_id": "u1"}])
        kwargs = collection.add.call_args.kwargs
        assert kwargs["ids"] == ["i1"]
        assert kwargs["documents"] == ["text"]
        assert kwargs["metadatas"] == [{"user_id": "u1"}]

    def test_supports_batch_writes(self, db, collection):
        db.add(ids=["a", "b"], embeddings=[[0.1], [0.2]],
               documents=["t1", "t2"], metadatas=[{}, {}])
        assert len(collection.add.call_args.kwargs["ids"]) == 2


class TestSearch:
    def _results(self, n=2):
        return {
            "documents": [[f"doc{i}" for i in range(n)]],
            "metadatas": [[{"user_id": "u1"} for _ in range(n)]],
            "distances": [[0.1 * i for i in range(n)]],
        }

    def test_flattens_chroma_lists_into_dicts(self, db, collection):
        """Note this is the OPPOSITE translation from the RAG store, which
        preserves the nested shape. Conversation-memory callers iterate
        results directly, so flattening happens here."""
        collection.query.return_value = self._results(2)
        results = db.search([0.1, 0.2])

        assert len(results) == 2
        assert results[0] == {"text": "doc0", "metadata": {"user_id": "u1"}, "score": 0.0}

    def test_result_keys_match_what_recall_callers_read(self, db, collection):
        collection.query.return_value = self._results(1)
        assert set(db.search([0.1])[0]) == {"text", "metadata", "score"}

    def test_wraps_the_embedding_in_a_batch_list(self, db, collection):
        collection.query.return_value = self._results(1)
        db.search([0.1, 0.2])
        assert collection.query.call_args.kwargs["query_embeddings"] == [[0.1, 0.2]]

    def test_default_top_k_is_five(self, db, collection):
        collection.query.return_value = self._results(1)
        db.search([0.1])
        assert collection.query.call_args.kwargs["n_results"] == 5

    def test_passes_filters_through_as_where(self, db, collection):
        """This is the only thing keeping one user's memories out of
        another user's recall."""
        collection.query.return_value = self._results(1)
        db.search([0.1], filters={"user_id": "u1"})
        assert collection.query.call_args.kwargs["where"] == {"user_id": "u1"}

    def test_no_filter_passes_none(self, db, collection):
        collection.query.return_value = self._results(1)
        db.search([0.1])
        assert collection.query.call_args.kwargs["where"] is None

    def test_empty_results_return_empty_list(self, db, collection):
        collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        assert db.search([0.1]) == []

    def test_preserves_chroma_ordering(self, db, collection):
        collection.query.return_value = self._results(3)
        assert [r["text"] for r in db.search([0.1])] == ["doc0", "doc1", "doc2"]


class TestDelete:
    def test_deletes_by_where_filter(self, db, collection):
        db.delete({"user_id": "u1"})
        collection.delete.assert_called_once_with(where={"user_id": "u1"})


class TestCreateConversationVectorStoreFactory:
    def test_default_provider_is_chroma(self, monkeypatch):
        monkeypatch.delenv("CHAT_VECTOR_STORE_PROVIDER", raising=False)
        with patch("src.memory.vectordb.Client"):
            assert isinstance(create_conversation_vector_store("c"), ChromaVectorDB)

    def test_env_var_selects_the_provider(self, monkeypatch):
        monkeypatch.setenv("CHAT_VECTOR_STORE_PROVIDER", "chroma")
        with patch("src.memory.vectordb.Client"):
            assert isinstance(create_conversation_vector_store("c"), ChromaVectorDB)

    def test_parameter_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("CHAT_VECTOR_STORE_PROVIDER", "azure_search")
        with patch("src.memory.vectordb.Client"):
            assert isinstance(create_conversation_vector_store("c", provider="chroma"), ChromaVectorDB)

    def test_provider_name_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("CHAT_VECTOR_STORE_PROVIDER", "  CHROMA  ")
        with patch("src.memory.vectordb.Client"):
            assert isinstance(create_conversation_vector_store("c"), ChromaVectorDB)

    def test_azure_search_is_not_implemented_yet(self, monkeypatch):
        """Step 4b. Must fail loudly rather than silently using Chroma —
        a silent fallback would write conversation memory to the wrong store."""
        monkeypatch.setenv("CHAT_VECTOR_STORE_PROVIDER", "azure_search")
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            create_conversation_vector_store("c")

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("CHAT_VECTOR_STORE_PROVIDER", "weaviate")
        with pytest.raises(ValueError, match="Unknown CHAT_VECTOR_STORE_PROVIDER"):
            create_conversation_vector_store("c")

    def test_this_switch_is_independent_of_the_rag_one(self, monkeypatch):
        """RAG chunks and conversation memory are two indexes by design.
        VECTOR_STORE_PROVIDER must not leak into this factory."""
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "azure_search")
        monkeypatch.delenv("CHAT_VECTOR_STORE_PROVIDER", raising=False)
        with patch("src.memory.vectordb.Client"):
            assert isinstance(create_conversation_vector_store("c"), ChromaVectorDB)
