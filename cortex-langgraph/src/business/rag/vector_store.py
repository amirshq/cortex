"""RAG vector store factory — LangChain edition.

cortex-core defined its own `VectorStoreBase` (reset/upsert/query) with two
hand-written backends, and the Azure one had to translate Azure Search's
response into Chroma's nested-list dict shape because retrieval.py parsed that
exact structure.

LangChain's `VectorStore` is a common interface, so none of that is needed:
both backends return `(Document, score)` pairs from
`similarity_search_with_score()`, and they embed text themselves given an
`Embeddings` object.

For production RAG systems you may prefer a managed vector DB. Chroma is the
default for simplicity; Azure AI Search is the cloud alternative.
"""

from __future__ import annotations

import os
from typing import Optional

from chromadb.config import Settings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

DEFAULT_COLLECTION = "pdf_chunks"

# Different default from cortex-core's "rag-chunks": LangChain's AzureSearch
# uses its own index schema (id, content, content_vector, metadata), which is
# not compatible with the index cortex-core creates. Distinct names mean the
# two projects can share one Azure Search service without clobbering each other.
DEFAULT_AZURE_INDEX = "rag-chunks-langchain"


# ── Provider selection ────────────────────────────────────────────────────
#
# VECTOR_STORE_PROVIDER (env var) picks the RAG vector store backend.
# Defaults to today's behavior — on-prem/local deployments don't need to
# set anything.
#
#   chroma        (default) — langchain_chroma.Chroma, local persistence.
#   azure_search  — langchain_azure_ai AzureSearch. Requires
#                   AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, and
#                   optionally AZURE_SEARCH_INDEX_NAME / AZURE_SEARCH_EMBEDDING_DIM.

def _provider(provider: Optional[str]) -> str:
    return (provider or os.getenv("VECTOR_STORE_PROVIDER", "chroma")).strip().lower()


def create_vector_store(
    persist_dir: str,
    collection_name: str = DEFAULT_COLLECTION,
    provider: Optional[str] = None,
    embedding: Optional[Embeddings] = None,
) -> VectorStore:
    """Factory for the RAG vector store, selected by VECTOR_STORE_PROVIDER or `provider`.

    embedding: the model used to embed chunks and queries. Defaults to
    config.yml → models.rag_embedding — it must be the same model at index and
    query time, so changing it requires re-indexing.
    """
    provider = _provider(provider)
    if provider not in ("chroma", "azure_search"):
        raise ValueError(
            f"Unknown VECTOR_STORE_PROVIDER={provider!r}. Supported: chroma, azure_search."
        )

    if embedding is None:
        from src.business.core.embedding import create_embedder

        embedding = create_embedder(role="rag_embedding")

    if provider == "chroma":
        return _chroma(persist_dir, collection_name, embedding)

    endpoint, api_key, index_name = _azure_settings()
    # Imported lazily so Chroma-only deployments never load the Azure SDK.
    from langchain_azure_ai.vectorstores import AzureSearch

    return AzureSearch(
        azure_search_endpoint=endpoint,
        azure_search_key=api_key,
        index_name=index_name,
        embedding_function=embedding,
        # Passing the dimension up front stops AzureSearch from embedding a
        # probe string (a paid API call) just to discover it. The index is
        # created on first use if it doesn't exist.
        vector_search_dimensions=int(os.getenv("AZURE_SEARCH_EMBEDDING_DIM", "1536")),
    )


def reset_vector_store(
    persist_dir: str,
    collection_name: str = DEFAULT_COLLECTION,
    provider: Optional[str] = None,
) -> None:
    """Wipe the RAG index to empty.

    Upserts never delete, so stale chunks from a previous upload would
    otherwise keep matching queries. Needs no embedding model.
    """
    provider = _provider(provider)

    if provider == "chroma":
        _chroma(persist_dir, collection_name, embedding=None).reset_collection()
        return

    if provider == "azure_search":
        endpoint, api_key, index_name = _azure_settings()
        from azure.core.credentials import AzureKeyCredential
        from azure.core.exceptions import ResourceNotFoundError
        from azure.search.documents.indexes import SearchIndexClient

        client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        try:
            # AzureSearch recreates the index on its next construction.
            client.delete_index(index_name)
        except ResourceNotFoundError:
            pass
        return

    raise ValueError(
        f"Unknown VECTOR_STORE_PROVIDER={provider!r}. Supported: chroma, azure_search."
    )


def _chroma(persist_dir: str, collection_name: str, embedding: Optional[Embeddings]) -> VectorStore:
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_dir,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True,
                                 persist_directory=persist_dir),
    )


def _azure_settings() -> tuple[str, str, str]:
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", DEFAULT_AZURE_INDEX)
    if not endpoint or not api_key:
        raise RuntimeError(
            "VECTOR_STORE_PROVIDER=azure_search requires AZURE_SEARCH_ENDPOINT "
            "and AZURE_SEARCH_API_KEY."
        )
    return endpoint, api_key, index_name
