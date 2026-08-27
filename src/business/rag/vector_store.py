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


# ── Provider selection ────────────────────────────────────────────────────
#
# VECTOR_STORE_PROVIDER (env var) picks the RAG vector store backend.
# Defaults to today's behavior — on-prem/local deployments don't need to
# set anything.
#
#   chroma        (default) — local Chroma persistence.
#   azure_search  — Azure AI Search (vector search). Not implemented yet.

def create_vector_store(
    persist_dir: str,
    collection_name: str = "pdf_chunks",
    provider: Optional[str] = None,
) -> VectorStoreBase:
    """Factory for the RAG vector store, selected by VECTOR_STORE_PROVIDER or `provider`."""
    provider = (provider or os.getenv("VECTOR_STORE_PROVIDER", "chroma")).strip().lower()

    if provider == "chroma":
        return ChromaVectorStore(persist_dir=persist_dir, collection_name=collection_name)

    if provider == "azure_search":
        raise NotImplementedError(
            "VECTOR_STORE_PROVIDER=azure_search is not implemented yet. Use 'chroma' for now."
        )

    raise ValueError(
        f"Unknown VECTOR_STORE_PROVIDER={provider!r}. Supported: chroma, "
        "azure_search (coming soon)."
    )
