import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from chromadb import Client
from chromadb.config import Settings

# This is the chatbot's conversation/long-term-memory vector store — a
# separate index from the RAG vector store (src/business/rag/vector_store.py).
# Kept separate by design: RAG chunks (PDF content) and per-user conversation
# history serve different purposes and shouldn't share an index.


class ConversationVectorStoreBase(ABC):
    """Interface every conversation/long-term-memory vector store backend must implement."""

    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]: ...

    @abstractmethod
    def delete(self, filters: Dict) -> None: ...


class ChromaVectorDB(ConversationVectorStoreBase):
    """
    Low-level vector database adapter.
    No AI logic. No memory semantics.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./chroma",
    ):
        self.client = Client(
            Settings(persist_directory=persist_directory)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
    ) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters,
        )

        return [
            {
                "text": doc,
                "metadata": meta,
                "score": score,
            }
            for doc, meta, score in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def delete(self, filters: Dict) -> None:
        self.collection.delete(where=filters)


# Backward-compat alias — existing code imports `VectorDB` directly.
VectorDB = ChromaVectorDB


# ── Provider selection ────────────────────────────────────────────────────
#
# CHAT_VECTOR_STORE_PROVIDER (env var) picks the conversation-memory vector
# store backend. This is intentionally a separate switch from RAG's
# VECTOR_STORE_PROVIDER — the two are different indexes by design. Defaults
# to today's behavior — on-prem/local deployments don't need to set anything.
#
#   chroma        (default) — local Chroma persistence.
#   azure_search  — Azure AI Search, a separate index from the RAG one. Not implemented yet.

def create_conversation_vector_store(
    collection_name: str,
    persist_directory: str = "./chroma",
    provider: Optional[str] = None,
) -> ConversationVectorStoreBase:
    """Factory for the conversation-memory vector store, selected by CHAT_VECTOR_STORE_PROVIDER."""
    provider = (provider or os.getenv("CHAT_VECTOR_STORE_PROVIDER", "chroma")).strip().lower()

    if provider == "chroma":
        return ChromaVectorDB(collection_name=collection_name, persist_directory=persist_directory)

    if provider == "azure_search":
        raise NotImplementedError(
            "CHAT_VECTOR_STORE_PROVIDER=azure_search is not implemented yet. Use 'chroma' for now."
        )

    raise ValueError(
        f"Unknown CHAT_VECTOR_STORE_PROVIDER={provider!r}. Supported: chroma, "
        "azure_search (coming soon)."
    )
