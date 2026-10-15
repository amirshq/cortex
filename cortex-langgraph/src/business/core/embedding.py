"""Embedding factory — LangChain edition.

cortex-core defined its own `Embedder` ABC and an `OpenAIEmbedder` with a
hand-rolled exponential-backoff retry loop. LangChain's `Embeddings` interface
has the same two methods (`embed_documents`, `embed_query`), and
`OpenAIEmbeddings` / `AzureOpenAIEmbeddings` implement it with retries built in.

Because every LangChain vector store takes an `Embeddings` object directly,
the vector stores now embed text themselves — callers hand them strings /
Documents instead of pre-computed vectors.

New compared to cortex-core: an optional on-disk embedding cache
(`CacheBackedEmbeddings`). Re-indexing the same PDF re-embeds identical chunks;
with the cache those come from disk instead of a paid API call.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from langchain_core.embeddings import Embeddings

from src.utils.config import model_settings

from .model import azure_openai_settings


# ── Provider selection ────────────────────────────────────────────────────
#
# EMBEDDING_PROVIDER (env var) picks the embedding implementation. Defaults
# to today's behavior — on-prem/local deployments don't need to set anything.
#
#   openai        (default) — OpenAIEmbeddings; model from config.yml →
#                   models.embedding.name.
#   azure_openai  — AzureOpenAIEmbeddings. Requires AZURE_OPENAI_ENDPOINT,
#                   AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME.

def create_embedder(
    provider: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    role: str = "embedding",
    cache_dir: Optional[str] = None,
) -> Embeddings:
    """Factory for the embedding model, selected by EMBEDDING_PROVIDER or `provider`.

    role: which entry of config.yml's `models:` names the model —
          "embedding" (chat memory, the default) or "rag_embedding" (the RAG index).
          They are separate because each index must be queried with the model it
          was built with, and the two indexes are rebuilt at different times.

    cache_dir: when given, wraps the model in a CacheBackedEmbeddings backed by
    a local file store in that directory. Off by default so tests and one-off
    scripts never write cache files unless asked to.
    """
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "openai")).strip().lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY must be set for EMBEDDING_PROVIDER=openai")
        model_name = model or model_settings(role)["name"]
        embedder: Embeddings = OpenAIEmbeddings(model=model_name, api_key=resolved_key, max_retries=5)

    elif provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        deployment = model or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        if not deployment:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=azure_openai requires "
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME (the deployment name you "
                "gave the embedding model in Azure — not the underlying model name)."
            )
        endpoint, resolved_key, api_version = azure_openai_settings(api_key)
        model_name = deployment
        embedder = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=resolved_key,
            api_version=api_version,
            azure_deployment=deployment,
            max_retries=5,
        )

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER={provider!r}. Supported: openai, azure_openai."
        )

    if cache_dir:
        return with_embedding_cache(embedder, cache_dir=cache_dir, namespace=f"{provider}:{model_name}")
    return embedder


def with_embedding_cache(embedder: Embeddings, *, cache_dir: str, namespace: str) -> Embeddings:
    """Wrap an embedder so repeated texts are served from a local file cache.

    The namespace includes provider + model: vectors from different models are
    not interchangeable, so they must never share cache entries.
    query_embedding_cache=True caches query embeddings too (same store), so a
    repeated question skips the embedding call as well.
    """
    from langchain_classic.embeddings import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore

    # The namespace becomes part of every cache key, and LocalFileStore only
    # accepts [A-Za-z0-9_.-/] in keys — "openai:text-embedding-3-small" would
    # be rejected on the first write. Replace anything else (and "/", which
    # would create subdirectories) with "_".
    safe_namespace = re.sub(r"[^A-Za-z0-9_.\-]", "_", namespace)

    store = LocalFileStore(cache_dir)
    return CacheBackedEmbeddings.from_bytes_store(
        embedder,
        store,
        namespace=safe_namespace,
        query_embedding_cache=True,
        # Keys are a hash of the text. The default (SHA-1) isn't
        # collision-resistant and LangChain warns about it; SHA-256 is.
        key_encoder="sha256",
    )
