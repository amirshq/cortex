"""End-to-end RAG workflows: HTTP in, real query_rag / ingest_pdfs behind it.

query_rag is not a pass-through: it decides the confidence signal (reranked
top chunk vs. hybrid fallback), trims source text, picks which score to
report, and records three metrics. ingest_pdfs resets and rebuilds the index.
The integration tier mocks both functions away; these tests keep them real.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

pytestmark = pytest.mark.e2e

QUERIES = "rag_queries_total"
TOP_SCORE_COUNT = "rag_retrieval_top_score_count"
LOW_CONFIDENCE = "rag_retrieval_low_confidence_total"
DOCS_INDEXED = "rag_documents_indexed_total"
CHUNKS_INDEXED = "rag_chunks_indexed_total"


def ask(client, question="How long is the warranty?"):
    return client.post("/api/v1/rag/query", json={"question": question})


def upload(client, name="manual.pdf", content=b"%PDF-1.4\n% e2e\n", content_type="application/pdf"):
    return client.post("/api/v1/rag/upload", files={"file": (name, content, content_type)})


class TestRagQuery:
    def test_a_confident_answer_returns_trimmed_sources_scored_by_the_reranker(
            self, client, rag_pipeline, metric):
        long_text = "Warranty terms apply. " * 50
        rag_pipeline.answer = "The warranty is 24 months."
        rag_pipeline.chunks = [Document(
            id="c1", page_content=long_text,
            metadata={"source_id": "manual.pdf", "section": "text",
                      "vector_score": 0.12, "rerank_score": 4.567891})]
        rag_pipeline.confidence = "high"
        queries, top, low = metric(QUERIES), metric(TOP_SCORE_COUNT), metric(LOW_CONFIDENCE)

        response = ask(client)

        assert response.status_code == 200
        assert response.json() == {
            "answer": "The warranty is 24 months.",
            "sources": [{"text": long_text[:400],
                         "metadata": {"source_id": "manual.pdf", "section": "text"},
                         "score": 4.5679}],
        }
        assert rag_pipeline.questions == ["How long is the warranty?"]
        assert (metric(QUERIES) - queries, metric(TOP_SCORE_COUNT) - top, metric(LOW_CONFIDENCE) - low) == (1, 1, 0)

    def test_a_gated_fallback_is_scored_by_vector_distance_and_flagged_low_confidence(
            self, client, rag_pipeline, metric):
        rag_pipeline.answer = "Europe uses 868 MHz."
        rag_pipeline.chunks = [Document(
            id="c2", page_content="EU frequency: 868 MHz.",
            metadata={"source_id": "specs.pdf", "vector_score": 0.3141592})]
        rag_pipeline.confidence = "low"
        queries, top, low = metric(QUERIES), metric(TOP_SCORE_COUNT), metric(LOW_CONFIDENCE)

        body = ask(client, "What radio frequency is used in Europe?").json()

        assert body["sources"] == [{"text": "EU frequency: 868 MHz.",
                                    "metadata": {"source_id": "specs.pdf"}, "score": 0.3142}]
        assert (metric(QUERIES) - queries, metric(TOP_SCORE_COUNT) - top, metric(LOW_CONFIDENCE) - low) == (1, 0, 1)

    def test_no_retrieved_context_still_answers_and_counts_as_low_confidence(
            self, client, rag_pipeline, metric):
        rag_pipeline.answer = "I don't have that information."
        low = metric(LOW_CONFIDENCE)

        body = ask(client).json()

        assert body == {"answer": "I don't have that information.", "sources": []}
        assert metric(LOW_CONFIDENCE) - low == 1

    def test_the_pipeline_opens_its_index_inside_the_sandbox(self, client, rag_pipeline, sandboxed_paths):
        ask(client)

        (persist_dir,) = rag_pipeline.constructed_with
        assert Path(persist_dir).is_relative_to(sandboxed_paths.project_root)

    def test_a_pipeline_failure_is_a_500_and_is_not_counted_as_a_query(self, client, rag_pipeline, metric):
        rag_pipeline.error = RuntimeError("vector store unavailable")
        queries = metric(QUERIES)

        response = ask(client)

        assert response.status_code == 500
        assert response.json() == {"detail": "RAG query failed: vector store unavailable"}
        assert metric(QUERIES) - queries == 0


class TestRagUpload:
    def test_a_valid_pdf_is_saved_then_the_index_is_reset_and_rebuilt(
            self, client, rag_index, sandboxed_paths, metric):
        content = b"%PDF-1.4\n% e2e manual\n"
        docs, chunks = metric(DOCS_INDEXED), metric(CHUNKS_INDEXED)

        response = upload(client, content=content)

        assert response.status_code == 200
        assert response.json() == {"filename": "manual.pdf", "docs_indexed": 1, "chunks_indexed": 7,
                                   "table_ocr_enabled": True, "message": "PDF indexed successfully"}
        assert (sandboxed_paths.uploads_dir / "manual.pdf").read_bytes() == content
        assert [e for e in rag_index.events if e in ("reset", "build")] == ["reset", "build"]
        (build,) = rag_index.build_calls
        assert build["data_dir"] == sandboxed_paths.uploads_dir
        assert build["pdfs"] == ["manual.pdf"]
        assert build["persist_dir"].is_relative_to(sandboxed_paths.project_root)
        assert rag_index.chunks == [f"chunk {i}" for i in range(7)]
        assert (metric(DOCS_INDEXED) - docs, metric(CHUNKS_INDEXED) - chunks) == (1, 7)

    def test_a_second_upload_indexes_only_the_new_pdf(self, client, rag_index, sandboxed_paths):
        upload(client, "first.pdf")
        upload(client, "second.pdf")

        assert [call["pdfs"] for call in rag_index.build_calls] == [["first.pdf"], ["second.pdf"]]
        assert sorted(p.name for p in sandboxed_paths.uploads_dir.iterdir()) == ["second.pdf"]

    def test_a_rejected_file_type_touches_neither_disk_nor_index(self, client, rag_index, sandboxed_paths):
        response = upload(client, "notes.txt", b"plain text", "text/plain")

        assert response.status_code == 400
        assert rag_index.events == []
        assert rag_index.chunks == ["chunk from a previous upload"]
        assert list(sandboxed_paths.uploads_dir.iterdir()) == []

    def test_an_empty_filename_is_refused_without_indexing(self, client, rag_index):
        response = upload(client, "")

        assert 400 <= response.status_code < 500
        assert rag_index.events == []

    def test_an_indexing_failure_is_a_500_with_no_indexing_metrics(self, client, rag_index, metric):
        rag_index.build_error = RuntimeError("docling crashed")
        docs, chunks = metric(DOCS_INDEXED), metric(CHUNKS_INDEXED)

        response = upload(client)

        assert response.status_code == 500
        assert response.json() == {"detail": "PDF upload failed: docling crashed"}
        assert (metric(DOCS_INDEXED) - docs, metric(CHUNKS_INDEXED) - chunks) == (0, 0)


class TestKnownIssueFailedUploadDestroysExistingContent:
    """KNOWN BUG. Upload deletes every previous PDF before ingesting, and
    ingest_pdfs resets the index before rebuilding it. If the rebuild fails,
    the user is left with no PDFs and an empty index — a failed upload
    destroys content that was working a moment earlier.
    """

    @pytest.mark.xfail(strict=True, reason="KNOWN BUG: a failed upload wipes the previous PDFs and index")
    def test_a_failed_upload_keeps_the_previously_indexed_content(self, client, rag_index, sandboxed_paths):
        previous = sandboxed_paths.uploads_dir / "previous.pdf"
        previous.write_bytes(b"%PDF-1.4 previous")
        rag_index.build_error = RuntimeError("docling crashed")

        assert upload(client, "broken.pdf").status_code == 500

        assert rag_index.chunks == ["chunk from a previous upload"]
        assert previous.exists()
