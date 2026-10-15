"""Conversation-memory vector store — LangChain edition.

This is the chatbot's long-term-memory index — a separate index from the RAG
vector store (src/business/rag/vector_store.py). Kept separate by design: RAG
chunks (PDF content) and per-user conversation history serve different
purposes and shouldn't share an index.

cortex-core wrapped chromadb by hand in a `ChromaVectorDB` adapter with its own
`add/search/delete` interface. LangChain's `VectorStore` is that interface, and
`langchain_chroma.Chroma` implements it — including embedding the text itself,
so callers pass strings instead of pre-computed vectors.

Behavior note: cortex-core created `chromadb.Client(Settings(persist_directory=...))`,
which in chromadb 1.x is an in-memory client — the directory was never written
and long-term memory was lost on every restart. Chroma(persist_directory=...)
here really persists, so memories now survive restarts.
"""

import os
from typing import Optional

from chromadb.config import Settings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


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
    embedding: Embeddings,
    persist_directory: str = "./chroma",
    provider: Optional[str] = None,
) -> VectorStore:
    """Factory for the conversation-memory vector store, selected by CHAT_VECTOR_STORE_PROVIDER."""
    provider = (provider or os.getenv("CHAT_VECTOR_STORE_PROVIDER", "chroma")).strip().lower()

    if provider == "chroma":
        from langchain_chroma import Chroma

        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            # Chroma's product telemetry would otherwise phone home on startup.
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True,
                                     persist_directory=persist_directory),
        )

    if provider == "azure_search":
        raise NotImplementedError(
            "CHAT_VECTOR_STORE_PROVIDER=azure_search is not implemented yet. Use 'chroma' for now."
        )

    raise ValueError(
        f"Unknown CHAT_VECTOR_STORE_PROVIDER={provider!r}. Supported: chroma, "
        "azure_search (coming soon)."
    )
