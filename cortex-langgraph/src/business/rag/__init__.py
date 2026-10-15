"""
RAG pipeline business logic entry-points.

query_rag              → called by RAGController.query (saves the Q&A when a user_id is given)
ingest_pdfs            → called by RAGController.upload (after file is saved to uploads dir)
list_indexed_documents → which PDFs are in the index (survives restarts)
get_rag_history / clear_rag_history → a user's past PDF Q&A

The pipeline and the indexer are synchronous and slow (an LLM call; minutes of
PDF parsing). They run in a worker thread via asyncio.to_thread, so the server
keeps answering other requests — chat included — while a PDF is processed.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from src.api.metrics import (
    RAG_QUERIES_TOTAL,
    RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL,
    RAG_RETRIEVAL_TOP_SCORE,
)
from src.business.rag.retrieval import RAGPipeline
from src.business.rag.index_builder import MANIFEST_NAME, build_index, read_manifest
from src.business.rag.vector_store import reset_vector_store
from src.memory.rag_history import RagHistoryManager
from src.utils.config import load_config

load_dotenv()

# Docling extracts table structure locally (no external API/key needed), so
# table extraction is always available.
TABLE_OCR_ENABLED: bool = True

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Metadata keys the graph adds for its own bookkeeping; not part of the
# chunk's provenance, so they're not echoed back in `sources[].metadata`.
_SCORE_KEYS = ("vector_score", "rerank_score")


def rag_persist_dir() -> Path:
    """Resolve the RAG vector-store location. Public (no leading underscore)
    because scripts/index_cli.py imports it too — the CLI and the API must
    always agree on where the index lives, or CLI-indexed content becomes
    invisible to /api/v1/rag/query and vice versa."""
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


def _history() -> RagHistoryManager:
    """Q&A history, in the same SQLite file as the chat history."""
    cfg = load_config()
    path = Path(os.getenv("SQLITE_DB_PATH") or cfg.get("directories", {}).get("db_path", "data/chatbot.db"))
    return RagHistoryManager(db_path=str(path if path.is_absolute() else _PROJECT_ROOT / path))


def _make_pipeline() -> RAGPipeline:
    return RAGPipeline(persist_dir=str(rag_persist_dir()))


async def query_rag(question: str, user_id: Optional[str] = None) -> Dict:
    """Run the RAG graph: retrieve → rerank → (gate) → generate.

    With a user_id, the question, answer and sources are saved, so the RAG
    panel can show past Q&A after a mode switch or a restart.
    """
    pipeline = _make_pipeline()
    answer_text, selected_chunks, confidence = await asyncio.to_thread(pipeline.answer, question)

    RAG_QUERIES_TOTAL.inc()
    # The graph reports which branch ran. "high" means the re-ranker found
    # something above its relevance gate; anything else means it fell back to
    # raw vector order (or refused) — itself the strongest signal that
    # retrieval quality is degrading.
    if confidence == "high" and selected_chunks:
        RAG_RETRIEVAL_TOP_SCORE.observe(selected_chunks[0].metadata["rerank_score"])
    elif confidence != "unranked":
        # "unranked" = no re-ranker configured: there is no gate, so no
        # confidence signal to report (not a low-confidence answer).
        RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL.inc()

    sources: List[Dict] = [
        {
            "text": chunk.page_content[:400],
            "metadata": {k: v for k, v in chunk.metadata.items() if k not in _SCORE_KEYS},
            # Re-ranked chunks carry rerank_score; fallback chunks only vector_score.
            "score": round(float(chunk.metadata.get("rerank_score", chunk.metadata.get("vector_score", 0.0))), 4),
        }
        for chunk in selected_chunks
    ]

    if user_id is not None:
        _history().save(user_id, question, answer_text, sources)

    return {"answer": answer_text, "sources": sources}


async def ingest_pdfs(uploads_dir: Path) -> Dict:
    """
    Build / rebuild the vector index from all PDFs in uploads_dir.
    The collection is reset first so stale chunks from previous uploads
    are removed before the new PDF is indexed.
    """
    persist_dir = rag_persist_dir()

    # Wipe old chunks — upsert never deletes, so stale content from previous
    # uploads would otherwise persist and pollute query results.
    reset_vector_store(persist_dir=str(persist_dir))
    # The manifest describes the old index; build_index writes a new one.
    (persist_dir / MANIFEST_NAME).unlink(missing_ok=True)

    docs_count, chunks_count = await asyncio.to_thread(
        build_index,
        data_dir=uploads_dir,
        persist_dir=persist_dir,
        # No truncation. The old 500,000-character budget silently dropped the
        # end of long books.
        max_context_chars=None,
    )
    return {
        "docs_indexed": docs_count,
        "chunks_indexed": chunks_count,
        "table_ocr_enabled": TABLE_OCR_ENABLED,
    }


def list_indexed_documents() -> List[Dict]:
    """PDFs currently in the RAG index: [{source_id, pages, chunks, indexed_at}]."""
    return read_manifest(rag_persist_dir())


def get_rag_history(user_id: str) -> List[Dict]:
    """A user's past PDF Q&A, oldest first."""
    return _history().list(user_id)


def clear_rag_history(user_id: str) -> int:
    """Delete a user's PDF Q&A. Returns how many entries were removed."""
    return _history().clear(user_id)
