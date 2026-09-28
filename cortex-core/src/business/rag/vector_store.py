"""Chroma vector store wrapper.

For production RAG/chatbot systems you may prefer managed vector DBs
like Weaviate or Pinecone. Chroma is used here for simplicity; swap it
out for scalability/performance in production.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions


class VectorStoreBase(ABC):
    """Interface every RAG vector store backend must implement."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> None: ...

    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int = 15): ...


class ChromaVectorStore(VectorStoreBase):
    def __init__(self, persist_dir: str, collection_name: str = "pdf_chunks", dim: int | None = None):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # we supply embeddings manually
        )
        self.dim = dim

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(self.collection.name)

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ):
        # check the lengths matches for ids, embeddings, metadatas, documents
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings length mismatch")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
    # Run the similarity search and return top_k results
    def query(self, query_embedding: List[float], top_k: int = 15):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )


# Backward-compat alias — existing code imports `VectorStore` directly.
VectorStore = ChromaVectorStore


class AzureSearchVectorStore(VectorStoreBase):
    """Azure AI Search-backed vector store for RAG chunks — same interface as
    ChromaVectorStore, backed by a cloud vector index instead of local Chroma.

    Creates the index on first use if it doesn't already exist yet. reset()
    drops and recreates the index, matching ChromaVectorStore.reset()'s
    "wipe to an empty collection" semantics exactly.

    The `azure-search-documents` SDK is imported lazily (inside methods, not
    at module level) so on-prem/Chroma-only deployments never need it installed.
    """

    INDEX_FIELDS_METADATA_KEYS = ("source_id", "section", "chunk_start", "chunk_end", "chunk_strategy")

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str = "rag-chunks",
        dim: int = 1536,
    ):
        from azure.core.credentials import AzureKeyCredential
        from azure.core.exceptions import ResourceNotFoundError
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        self._ResourceNotFoundError = ResourceNotFoundError
        self._SearchIndex = SearchIndex
        self.endpoint = endpoint
        self.index_name = index_name
        self.dim = dim

        credential = AzureKeyCredential(api_key)
        self._index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        self._search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

        algorithm_name = "rag-hnsw"
        profile_name = "rag-vector-profile"
        self._vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name=algorithm_name)],
            profiles=[VectorSearchProfile(name=profile_name, algorithm_configuration_name=algorithm_name)],
        )
        self._fields = [
            SimpleField(name="id", type=SearchFieldDataType.STRING, key=True),
            SearchField(name="content", type=SearchFieldDataType.STRING, searchable=True),
            SearchField(
                name="embedding",
                type="Collection(Edm.Single)",
                searchable=True,
                vector_search_dimensions=dim,
                vector_search_profile_name=profile_name,
            ),
            SimpleField(name="source_id", type=SearchFieldDataType.STRING, filterable=True),
            SimpleField(name="section", type=SearchFieldDataType.STRING, filterable=True),
            SimpleField(name="chunk_start", type=SearchFieldDataType.INT32, filterable=True),
            SimpleField(name="chunk_end", type=SearchFieldDataType.INT32, filterable=True),
            SimpleField(name="chunk_strategy", type=SearchFieldDataType.STRING, filterable=True),
        ]

        self._ensure_index_exists()

    def _ensure_index_exists(self) -> None:
        try:
            self._index_client.get_index(self.index_name)
        except self._ResourceNotFoundError:
            self._index_client.create_index(
                self._SearchIndex(name=self.index_name, fields=self._fields, vector_search=self._vector_search)
            )

    def reset(self) -> None:
        try:
            self._index_client.delete_index(self.index_name)
        except self._ResourceNotFoundError:
            pass
        self._index_client.create_index(
            self._SearchIndex(name=self.index_name, fields=self._fields, vector_search=self._vector_search)
        )

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> None:
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings length mismatch")

        docs = []
        for doc_id, embedding, metadata, text in zip(ids, embeddings, metadatas, documents):
            metadata = metadata or {}
            docs.append({
                "id": doc_id,
                "content": text,
                "embedding": embedding,
                "source_id": str(metadata.get("source_id", "")),
                "section": str(metadata.get("section", "")),
                "chunk_start": int(metadata.get("chunk_start", 0)),
                "chunk_end": int(metadata.get("chunk_end", 0)),
                "chunk_strategy": str(metadata.get("chunk_strategy", "")),
            })
        self._search_client.merge_or_upload_documents(documents=docs)

    def query(self, query_embedding: List[float], top_k: int = 15):
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=top_k, fields="embedding")
        results = self._search_client.search(search_text=None, vector_queries=[vector_query], top=top_k)

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        scores: List[float] = []
        for r in results:
            ids.append(r["id"])
            docs.append(r.get("content", ""))
            metas.append({key: r.get(key) for key in self.INDEX_FIELDS_METADATA_KEYS})
            # Azure Search's vector-query score is a similarity score (higher =
            # better); Chroma's "distances" are a distance (lower = better).
            # Negate it so "lower is better" holds for both backends — nothing
            # downstream treats this as a calibrated distance, it's only ever
            # used for display and (already-sorted) ordering.
            scores.append(-float(r.get("@search.score", 0.0)))

        # Match ChromaVectorStore.query()'s response shape exactly (outer list
        # = batch of queries; we only ever send one) since retrieval.py's
        # _retrieve() depends on this exact structure regardless of backend.
        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [scores],
        }


# ── Provider selection ────────────────────────────────────────────────────
#
# VECTOR_STORE_PROVIDER (env var) picks the RAG vector store backend.
# Defaults to today's behavior — on-prem/local deployments don't need to
# set anything.
#
#   chroma        (default) — local Chroma persistence.
#   azure_search  — Azure AI Search (vector search). Requires
#                   AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, and
#                   optionally AZURE_SEARCH_INDEX_NAME (see .env).

def create_vector_store(
    persist_dir: str,
    collection_name: str = "pdf_chunks",
    provider: Optional[str] = None,
    dim: Optional[int] = None,
) -> VectorStoreBase:
    """Factory for the RAG vector store, selected by VECTOR_STORE_PROVIDER or `provider`."""
    provider = (provider or os.getenv("VECTOR_STORE_PROVIDER", "chroma")).strip().lower()

    if provider == "chroma":
        return ChromaVectorStore(persist_dir=persist_dir, collection_name=collection_name)

    if provider == "azure_search":
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-chunks")
        if not endpoint or not api_key:
            raise RuntimeError(
                "VECTOR_STORE_PROVIDER=azure_search requires AZURE_SEARCH_ENDPOINT "
                "and AZURE_SEARCH_API_KEY."
            )
        resolved_dim = dim or int(os.getenv("AZURE_SEARCH_EMBEDDING_DIM", "1536"))
        return AzureSearchVectorStore(
            endpoint=endpoint, api_key=api_key, index_name=index_name, dim=resolved_dim
        )

    raise ValueError(
        f"Unknown VECTOR_STORE_PROVIDER={provider!r}. Supported: chroma, azure_search."
    )
