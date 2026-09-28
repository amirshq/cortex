"""Tests for query_rag / ingest_pdfs — the RAG business-layer entry points.

These are the functions the controller calls (and that test_controller.py
mocks away). They own two things nothing else tests: the retrieval-quality
metrics, and the "reset before re-index" rule that stops stale chunks from
a previous upload polluting query results.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.business.rag import ingest_pdfs, query_rag, rag_persist_dir
from src.business.rag.re_ranker.interface import ReRankedChunk, RetrievedChunk


def make_pipeline(answer="an answer", chunks=None):
    pipeline = MagicMock()
    pipeline.answer.return_value = (answer, chunks if chunks is not None else [])
    return pipeline


class TestQueryRag:
    @pytest.mark.asyncio
    async def test_returns_answer_and_sources(self):
        chunks = [ReRankedChunk("c1", "chunk text", {"source_id": "doc.pdf"}, 0.4, 0.87)]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)):
            result = await query_rag("q")

        assert result["answer"] == "A"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["metadata"] == {"source_id": "doc.pdf"}

    @pytest.mark.asyncio
    async def test_source_text_is_truncated_to_400_chars(self):
        """Sources go over the wire to the UI — full chunks would bloat it."""
        chunks = [ReRankedChunk("c1", "x" * 5000, {}, 0.4, 0.9)]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)):
            result = await query_rag("q")
        assert len(result["sources"][0]["text"]) == 400

    @pytest.mark.asyncio
    async def test_reranked_chunk_reports_rerank_score(self):
        chunks = [ReRankedChunk("c1", "t", {}, vector_score=0.42, rerank_score=0.87)]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)):
            result = await query_rag("q")
        assert result["sources"][0]["score"] == 0.87

    @pytest.mark.asyncio
    async def test_fallback_chunk_reports_vector_score_instead(self):
        """Low-confidence fallback yields plain RetrievedChunk — no rerank_score.

        getattr() must fall through to vector_score rather than raising.
        """
        chunks = [RetrievedChunk("c1", "t", {}, vector_score=0.42)]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)):
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
        chunks = [ReRankedChunk("c1", "t", {}, 0.4, rerank_score=0.91)]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)), \
             patch("src.business.rag.RAG_RETRIEVAL_TOP_SCORE") as histogram, \
             patch("src.business.rag.RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL") as low:
            await query_rag("q")

        histogram.observe.assert_called_once_with(0.91)
        low.inc.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_increments_the_low_confidence_counter(self):
        """The alert that says retrieval quality is degrading.

        A plain RetrievedChunk at position 0 means select_context fell back
        to raw vector order — that is the signal, so it must be counted and
        must NOT be observed into the score histogram.
        """
        chunks = [RetrievedChunk("c1", "t", {}, vector_score=0.4)]
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", chunks)), \
             patch("src.business.rag.RAG_RETRIEVAL_TOP_SCORE") as histogram, \
             patch("src.business.rag.RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL") as low:
            await query_rag("q")

        low.inc.assert_called_once()
        histogram.observe.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_chunks_counts_as_low_confidence(self):
        with patch("src.business.rag._make_pipeline", return_value=make_pipeline("A", [])), \
             patch("src.business.rag.RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL") as low:
            result = await query_rag("q")

        low.inc.assert_called_once()
        assert result["sources"] == []


class TestIngestPdfs:
    @pytest.mark.asyncio
    async def test_resets_the_collection_before_indexing(self):
        """upsert never deletes, so a re-upload without reset leaves stale
        chunks from the previous PDF answering queries about the new one."""
        store = MagicMock()
        with patch("src.business.rag.create_vector_store", return_value=store), \
             patch("src.business.rag.build_index", return_value=(1, 10)):
            await ingest_pdfs(Path("/tmp/uploads"))
        store.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_happens_before_build_index(self):
        """Ordering matters: reset after build would wipe the new index."""
        order = []
        store = MagicMock()
        store.reset.side_effect = lambda: order.append("reset")

        def fake_build(**kwargs):
            order.append("build")
            return (1, 10)

        with patch("src.business.rag.create_vector_store", return_value=store), \
             patch("src.business.rag.build_index", side_effect=fake_build):
            await ingest_pdfs(Path("/tmp/uploads"))

        assert order == ["reset", "build"]

    @pytest.mark.asyncio
    async def test_returns_index_counts(self):
        with patch("src.business.rag.create_vector_store"), \
             patch("src.business.rag.build_index", return_value=(3, 142)):
            result = await ingest_pdfs(Path("/tmp/uploads"))

        assert result["docs_indexed"] == 3
        assert result["chunks_indexed"] == 142
        assert result["table_ocr_enabled"] is True

    @pytest.mark.asyncio
    async def test_indexes_from_the_uploads_dir_it_was_given(self):
        with patch("src.business.rag.create_vector_store"), \
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
