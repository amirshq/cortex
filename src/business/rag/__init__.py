"""
RAG pipeline business logic entry-points.

query_rag   → called by RAGController.query
ingest_pdfs → called by RAGController.upload (after file is saved to uploads dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from src.business.rag.retrieval import RAGPipeline
from src.business.rag.index_builder import build_index
from src.business.rag.vector_store import VectorStore
from src.utils.config import load_config

load_dotenv()

# Docling extracts table structure locally (no external API/key needed), so
# table extraction is always available.
TABLE_OCR_ENABLED: bool = True

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _rag_persist_dir() -> Path:
    cfg = load_config()
    dirs = cfg.get("directories", {})
    path = _PROJECT_ROOT / dirs.get("rag_vectorstore_dir", "data/rag_vectorstore")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rag_uploads_dir() -> Path:
    cfg = load_config()
    dirs = cfg.get("directories", {})
    path = _PROJECT_ROOT / dirs.get("rag_uploads_dir", "data/rag_uploads")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_pipeline() -> RAGPipeline:
    return RAGPipeline(persist_dir=str(_rag_persist_dir()))


async def query_rag(question: str) -> Dict:
    """Run the full RAG pipeline: retrieve → rerank → generate."""
    pipeline = _make_pipeline()
    answer_text, selected_chunks = pipeline.answer(question)

    sources: List[Dict] = [
        {
            "text": chunk.text[:400],
            "metadata": chunk.metadata,
            # ReRankedChunk has rerank_score; fallback RetrievedChunk only has vector_score
            "score": round(float(getattr(chunk, "rerank_score", chunk.vector_score)), 4),
        }
        for chunk in selected_chunks
    ]

    return {"answer": answer_text, "sources": sources}


async def ingest_pdfs(uploads_dir: Path) -> Dict:
    """
    Build / rebuild the vector index from all PDFs in uploads_dir.
    The collection is reset first so stale chunks from previous uploads
    are removed before the new PDF is indexed.
    """
    persist_dir = _rag_persist_dir()

    # Wipe old chunks — upsert never deletes, so stale content from previous
    # uploads would otherwise persist and pollute query results.
    VectorStore(persist_dir=str(persist_dir)).reset()

    docs_count, chunks_count = build_index(
        data_dir=uploads_dir,
        persist_dir=persist_dir,
        max_context_chars=500_000,  # large enough to capture entire papers including tables
    )
    return {
        "docs_indexed": docs_count,
        "chunks_indexed": chunks_count,
        "table_ocr_enabled": TABLE_OCR_ENABLED,
    }
