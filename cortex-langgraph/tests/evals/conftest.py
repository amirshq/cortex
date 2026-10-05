"""Fixtures for the evaluation suite.

Everything here builds a REAL index with REAL embeddings, because that is
the only way retrieval quality means anything. The corpus is tiny (five
documents) so a full run costs a fraction of a cent.

Every fixture skips rather than fails when its prerequisite is missing —
an eval run without an API key should say "skipped", not "broken".
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List

import pytest
from dotenv import load_dotenv

# Load .env here rather than relying on some other module having imported
# src.business.rag (which calls load_dotenv() at import time) first.
# Without this the skip guard below depends on collection order: running
# the whole suite would find a key and running this directory alone would
# not, so the same tests would run for real or skip depending on the
# command line.
load_dotenv()

GOLDEN_SET_PATH = Path(__file__).parent / "data" / "golden_set.json"


@pytest.fixture(scope="session")
def golden_set() -> Dict:
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set — eval tests need a real provider")
    return key


@pytest.fixture(scope="session")
def real_embedder(require_openai_key):
    from src.business.core.embedding import create_embedder
    return create_embedder(api_key=require_openai_key)


def _index_corpus(persist_dir, collection_name, golden_set, embedder):
    """Real Chroma (LangChain) store with one chunk per golden document."""
    from langchain_core.documents import Document

    from src.business.rag.vector_store import create_vector_store

    store = create_vector_store(persist_dir=str(persist_dir), collection_name=collection_name,
                                provider="chroma", embedding=embedder)
    docs = golden_set["documents"]
    store.add_documents(
        [Document(page_content=d["text"], metadata={"source_id": d["id"], "section": "text"}) for d in docs],
        ids=[d["id"] for d in docs],
    )
    return store


@pytest.fixture(scope="session")
def indexed_corpus(tmp_path_factory, golden_set, real_embedder):
    """Build a real Chroma index over the golden corpus.

    One document per chunk (they are short), so a retrieved chunk maps 1:1
    back to a document id and recall@k is unambiguous.
    """
    return _index_corpus(tmp_path_factory.mktemp("eval_index"), "eval_chunks", golden_set, real_embedder)


class EvalReranker:
    """The production re-ranking step: CrossEncoderScorer + rerank_documents()."""

    def __init__(self, config):
        from src.business.rag.re_ranker.cross_encoder import CrossEncoderScorer

        self.config = config
        self.scorer = CrossEncoderScorer(config)

    def re_rank(self, query, documents):
        from src.business.rag.re_ranker.re_ranker import rerank_documents

        return rerank_documents(query, documents, self.scorer, self.config)


@pytest.fixture(scope="session")
def real_reranker():
    """Cross-encoder re-ranker. Skips if the model can't be loaded."""
    from src.business.rag.re_ranker.config import ReRankerConfig

    try:
        return EvalReranker(ReRankerConfig())
    except Exception as exc:  # noqa: BLE001 — model download / torch missing
        pytest.skip(f"cross-encoder re-ranker unavailable: {exc}")


@pytest.fixture(scope="session")
def selection_pipeline(indexed_corpus, real_reranker):
    """The REAL RAG graph over the golden corpus, with a free no-op LLM.

    Used to measure what the graph selects (retrieve → rerank → gate →
    fallback) on the production path without paying for generation. top_k_input
    is 5 to match the vector-only baselines these evals compare against.
    """
    import itertools
    from dataclasses import replace

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from src.business.rag.retrieval import RAGPipeline

    return RAGPipeline(
        persist_dir="unused",
        vector_store=indexed_corpus,
        scorer=real_reranker.scorer,
        reranker_config=replace(real_reranker.config, top_k_input=5),
        llm=GenericFakeChatModel(messages=itertools.repeat(AIMessage(content=""))),
    )


@pytest.fixture(scope="session")
def rag_pipeline(tmp_path_factory, golden_set, require_openai_key, real_embedder):
    """A full RAGPipeline wired to the golden corpus (real LLM, real cross-encoder)."""
    from src.business.rag.retrieval import RAGPipeline

    persist_dir = tmp_path_factory.mktemp("eval_pipeline_index")
    _index_corpus(persist_dir, "eval_pipeline", golden_set, real_embedder)

    try:
        pipeline = RAGPipeline(persist_dir=str(persist_dir), collection_name="eval_pipeline")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"RAGPipeline could not be constructed: {exc}")
    return pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def retrieve_docs(store, embedder, question: str, top_k: int):
    """Vector search → Documents (best first) with metadata["vector_score"],
    the same shape the graph's retrieve node produces."""
    from langchain_core.documents import Document

    hits = store.similarity_search_by_vector_with_relevance_scores(embedder.embed_query(question), k=top_k)
    return [Document(id=d.id, page_content=d.page_content,
                     metadata={**d.metadata, "vector_score": float(s)}) for d, s in hits]


def retrieve_doc_ids(store, embedder, question: str, top_k: int) -> List[str]:
    """Return retrieved document ids in rank order."""
    return [d.id for d in retrieve_docs(store, embedder, question, top_k)]


def select_context(pipeline, question: str):
    """Run the RAG graph; return (selected ids, confidence) — the production path."""
    state = pipeline.run(question)
    return [d.id for d in state["context"]], state["confidence"]


class Stopwatch:
    """Records elapsed wall-clock seconds for a labelled block."""

    def __init__(self, label: str):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile. Small samples, so no interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(p / 100.0 * len(ordered) + 0.5)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]
