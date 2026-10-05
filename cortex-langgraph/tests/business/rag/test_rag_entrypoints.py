"""Tests for query_rag / ingest_pdfs — the RAG business-layer entry points.

These are the functions the controller calls (and that test_controller.py
mocks away). They own two things nothing else tests: the retrieval-quality
metrics, and the "reset before re-index" rule that stops stale chunks from
a previous upload polluting query results.

In cortex-core the metrics inferred confidence from the chunk's Python type
(ReRankedChunk vs RetrievedChunk). The graph now reports it explicitly as
`confidence`, so these tests drive query_rag with (answer, chunks, confidence).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import threading

from src.business.rag import (
    clear_rag_history,
    get_rag_history,
    ingest_pdfs,
    list_indexed_documents,
    query_rag,
    rag_persist_dir,
)


def reranked(text="t", metadata=None, vector_score=0.4, rerank_score=0.87) -> Document:
    return Document(page_content=text,
                    metadata={**(metadata or {}), "vector_score": vector_score, "rerank_score": rerank_score})


def fallback(text="t", metadata=None, vector_score=0.42) -> Document:
    return Document(page_content=text, metadata={**(metadata or {}), "vector_score": vector_score})


def make_pipeline(answer="an answer", chunks=None, confidence="high"):
    pipeline = MagicMock()
    pipeline.answer.return_value = (answer, chunks if chunks is not None else [], confidence)
    return pipeline


class TestQueryRag:
    @pytest.mark.asyncio
    async def test_returns_answer_and_sources(self):
        chunks = [reranked("chunk text", {"source_id": "doc.pdf"})]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)):
            result = await query_rag("q")

        assert result["answer"] == "A"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["text"] == "chunk text"

    @pytest.mark.asyncio
    async def test_score_bookkeeping_is_not_echoed_as_metadata(self):
        """The graph's vector_score / rerank_score ride along in metadata; the
        API's `metadata` must stay the chunk's provenance, as in cortex-core."""
        chunks = [reranked("t", {"source_id": "doc.pdf", "section": "text"})]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)):
            result = await query_rag("q")
        assert result["sources"][0]["metadata"] == {"source_id": "doc.pdf", "section": "text"}

    @pytest.mark.asyncio
    async def test_source_text_is_truncated_to_400_chars(self):
        """Sources go over the wire to the UI — full chunks would bloat it."""
        chunks = [reranked("x" * 5000)]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)):
            result = await query_rag("q")
        assert len(result["sources"][0]["text"]) == 400

    @pytest.mark.asyncio
    async def test_reranked_chunk_reports_rerank_score(self):
        chunks = [reranked(vector_score=0.42, rerank_score=0.87)]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)):
            result = await query_rag("q")
        assert result["sources"][0]["score"] == 0.87

    @pytest.mark.asyncio
    async def test_fallback_chunk_reports_vector_score_instead(self):
        """Fallback chunks never went through the gate — no rerank_score."""
        chunks = [fallback(vector_score=0.42)]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks, "low")):
            result = await query_rag("q")
        assert result["sources"][0]["score"] == 0.42

    @pytest.mark.asyncio
    async def test_increments_the_query_counter(self):
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline()), \
             patch("src.business.rag.RAG_QUERIES_TOTAL") as counter:
            await query_rag("q")
        counter.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_high_confidence_observes_the_top_score_histogram(self):
        chunks = [reranked(rerank_score=0.91), reranked(rerank_score=0.5)]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks, "high")), \
             patch("src.business.rag.RAG_RETRIEVAL_TOP_SCORE") as histogram, \
             patch("src.business.rag.RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL") as low:
            await query_rag("q")

        histogram.observe.assert_called_once_with(0.91)
        low.inc.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_increments_the_low_confidence_counter(self):
        """The alert that says retrieval quality is degrading: the gate rejected
        everything and the graph took the fallback branch. It must be counted
        and must NOT be observed into the score histogram."""
        chunks = [fallback()]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks, "low")), \
             patch("src.business.rag.RAG_RETRIEVAL_TOP_SCORE") as histogram, \
             patch("src.business.rag.RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL") as low:
            await query_rag("q")

        low.inc.assert_called_once()
        histogram.observe.assert_not_called()

    @pytest.mark.asyncio
    async def test_refusal_counts_as_low_confidence(self):
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("I don't know.", [], "none")), \
             patch("src.business.rag.RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL") as low:
            result = await query_rag("q")

        low.inc.assert_called_once()
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_no_chunks_counts_as_low_confidence(self):
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", [], "high")), \
             patch("src.business.rag.RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL") as low:
            result = await query_rag("q")

        low.inc.assert_called_once()
        assert result["sources"] == []


class TestIngestPdfs:
    @pytest.mark.asyncio
    async def test_resets_the_collection_before_indexing(self):
        """upsert never deletes, so a re-upload without reset leaves stale
        chunks from the previous PDF answering queries about the new one."""
        with patch("src.business.rag.reset_vector_store") as reset, \
             patch("src.business.rag.build_index", return_value=(1, 10)):
            await ingest_pdfs(Path("/tmp/uploads"))
        reset.assert_called_once_with(persist_dir=str(rag_persist_dir()))

    @pytest.mark.asyncio
    async def test_reset_happens_before_build_index(self):
        """Ordering matters: reset after build would wipe the new index."""
        order = []

        def fake_build(**kwargs):
            order.append("build")
            return (1, 10)

        with patch("src.business.rag.reset_vector_store", side_effect=lambda **kw: order.append("reset")), \
             patch("src.business.rag.build_index", side_effect=fake_build):
            await ingest_pdfs(Path("/tmp/uploads"))

        assert order == ["reset", "build"]

    @pytest.mark.asyncio
    async def test_returns_index_counts(self):
        with patch("src.business.rag.reset_vector_store"), \
             patch("src.business.rag.build_index", return_value=(3, 142)):
            result = await ingest_pdfs(Path("/tmp/uploads"))

        assert result["docs_indexed"] == 3
        assert result["chunks_indexed"] == 142
        assert result["table_ocr_enabled"] is True

    @pytest.mark.asyncio
    async def test_indexes_from_the_uploads_dir_it_was_given(self):
        with patch("src.business.rag.reset_vector_store"), \
             patch("src.business.rag.build_index", return_value=(0, 0)) as build:
            await ingest_pdfs(Path("/tmp/specific-uploads"))

        assert build.call_args.kwargs["data_dir"] == Path("/tmp/specific-uploads")


class TestPersistDir:
    def test_cli_and_api_resolve_to_the_same_index_location(self):
        """scripts/index_cli.py imports this too. If the two disagreed,
        CLI-indexed content would be invisible to /api/v1/rag/query."""
        assert rag_persist_dir() == rag_persist_dir()

    def test_persist_dir_is_created(self):
        assert rag_persist_dir().exists()


class TestRagHistoryWiring:
    @pytest.fixture(autouse=True)
    def sandbox_db(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "chatbot.db"))

    @pytest.mark.asyncio
    async def test_query_with_a_user_is_saved(self):
        chunks = [reranked("chunk", {"source_id": "book.pdf"})]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)):
            await query_rag("What is RAG?", user_id="1")
        (item,) = get_rag_history("1")
        assert item["question"] == "What is RAG?" and item["answer"] == "A"
        assert item["sources"][0]["metadata"] == {"source_id": "book.pdf"}

    @pytest.mark.asyncio
    async def test_query_without_a_user_is_not_saved(self):
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", [])):
            await query_rag("anonymous question")
        assert get_rag_history("1") == []

    @pytest.mark.asyncio
    async def test_clear(self):
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", [])):
            await query_rag("q", user_id="1")
        assert clear_rag_history("1") == 1 and get_rag_history("1") == []


class TestNonBlocking:
    @pytest.mark.asyncio
    async def test_pipeline_runs_off_the_event_loop(self):
        """A RAG answer (or a 10-minute PDF ingest) must not freeze the server:
        both run in a worker thread, not on the event loop's thread."""
        seen = {}
        pipeline = MagicMock()

        def answer(question):
            seen["thread"] = threading.current_thread()
            return "A", [], "high"

        pipeline.answer.side_effect = answer
        with patch("src.business.rag._make_pipeline", return_value=pipeline):
            await query_rag("q")
        assert seen["thread"] is not threading.main_thread()

    @pytest.mark.asyncio
    async def test_ingest_runs_off_the_event_loop_and_is_not_truncated(self):
        seen = {}

        def fake_build(**kwargs):
            seen["thread"] = threading.current_thread()
            seen["max_context_chars"] = kwargs["max_context_chars"]
            return (1, 10)

        with patch("src.business.rag.reset_vector_store"), \
             patch("src.business.rag.build_index", side_effect=fake_build):
            await ingest_pdfs(Path("/tmp/uploads"))
        assert seen["thread"] is not threading.main_thread()
        # The old 500,000-character cap silently dropped the end of long books.
        assert seen["max_context_chars"] is None


class TestIndexedDocuments:
    def test_lists_the_manifest_written_by_build_index(self, monkeypatch, tmp_path):
        from langchain_core.documents import Document

        from src.business.rag.index_builder import write_manifest

        monkeypatch.setattr("src.business.rag.rag_persist_dir", lambda: tmp_path)
        docs = [Document(page_content="x", metadata={"source_id": "book.pdf", "page_count": 535})]
        chunks = [Document(page_content=str(i), metadata={"source_id": "book.pdf"}) for i in range(3)]
        write_manifest(tmp_path, docs, chunks)

        (doc,) = list_indexed_documents()
        assert (doc["source_id"], doc["pages"], doc["chunks"]) == ("book.pdf", 535, 3)

    def test_empty_index_lists_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.business.rag.rag_persist_dir", lambda: tmp_path)
        assert list_indexed_documents() == []

    @pytest.mark.asyncio
    async def test_reupload_removes_the_old_manifest_before_rebuilding(self, monkeypatch, tmp_path):
        """If the rebuild fails, the UI must not keep listing the old PDF as indexed."""
        (tmp_path / "documents.json").write_text('[{"source_id": "old.pdf", "chunks": 5}]')
        monkeypatch.setattr("src.business.rag.rag_persist_dir", lambda: tmp_path)
        with patch("src.business.rag.reset_vector_store"), \
             patch("src.business.rag.build_index", side_effect=RuntimeError("docling crashed")):
            with pytest.raises(RuntimeError):
                await ingest_pdfs(Path("/tmp/uploads"))
        assert list_indexed_documents() == []
