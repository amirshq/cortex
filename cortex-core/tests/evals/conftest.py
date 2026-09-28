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


@pytest.fixture(scope="session")
def indexed_corpus(tmp_path_factory, golden_set, real_embedder):
    """Build a real Chroma index over the golden corpus.

    One document per chunk (they are short), so a retrieved chunk maps 1:1
    back to a document id and recall@k is unambiguous.
    """
    from src.business.rag.vector_store import ChromaVectorStore

    persist_dir = tmp_path_factory.mktemp("eval_index")
    store = ChromaVectorStore(persist_dir=str(persist_dir), collection_name="eval_chunks")

    docs = golden_set["documents"]
    texts = [d["text"] for d in docs]
    embeddings = real_embedder.embed_documents(texts)

    store.upsert(
        ids=[d["id"] for d in docs],
        embeddings=embeddings,
        metadatas=[{"source_id": d["id"], "section": "text"} for d in docs],
        documents=texts,
    )
    return store


@pytest.fixture(scope="session")
def real_reranker():
    """Cross-encoder re-ranker. Skips if the model can't be loaded."""
    from src.business.rag.re_ranker.config import ReRankerConfig
    from src.business.rag.re_ranker.re_ranker import ReRanker

    try:
        from src.business.rag.re_ranker.cross_encoder import CrossEncoderReRanker
        config = ReRankerConfig()
        return ReRanker(scorer=CrossEncoderReRanker(config), config=config)
    except Exception as exc:  # noqa: BLE001 — model download / torch missing
        pytest.skip(f"cross-encoder re-ranker unavailable: {exc}")


@pytest.fixture(scope="session")
def rag_pipeline(tmp_path_factory, golden_set, require_openai_key, real_embedder):
    """A full RAGPipeline wired to the golden corpus."""
    from src.business.rag.retrieval import RAGPipeline
    from src.business.rag.vector_store import ChromaVectorStore

    persist_dir = tmp_path_factory.mktemp("eval_pipeline_index")
    store = ChromaVectorStore(persist_dir=str(persist_dir), collection_name="eval_pipeline")

    docs = golden_set["documents"]
    texts = [d["text"] for d in docs]
    store.upsert(
        ids=[d["id"] for d in docs],
        embeddings=real_embedder.embed_documents(texts),
        metadatas=[{"source_id": d["id"], "section": "text"} for d in docs],
        documents=texts,
    )

    try:
        pipeline = RAGPipeline(persist_dir=str(persist_dir), collection_name="eval_pipeline")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"RAGPipeline could not be constructed: {exc}")
    return pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def retrieve_doc_ids(store, embedder, question: str, top_k: int) -> List[str]:
    """Return retrieved document ids in rank order."""
    result = store.query(embedder.embed_query(question), top_k)
    return result.get("ids", [[]])[0]


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
