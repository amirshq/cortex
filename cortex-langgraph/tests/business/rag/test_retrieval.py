"""Tests for RAGPipeline — the retrieve → rerank → generate chain.

The re-ranker's scoring maths already has tests; what was missing is the
wiring around it: that the vector store's response shape is unpacked
correctly into RetrievedChunk, that the LLM is handed re-ranked text (not
raw vector order), and that a degenerate retrieval still produces a
response instead of an exception.

RAGPipeline.__init__ builds a CrossEncoderReRanker, which downloads a
transformer model. Every test here patches the constructor's collaborators
so nothing is fetched and no API key is required.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from src.business.rag.retrieval import RAGPipeline
from src.business.rag.re_ranker.config import ReRankerConfig
from src.business.rag.re_ranker.interface import ReRankedChunk, RetrievedChunk


def build_pipeline(vector_response=None, llm_answer="generated answer", config=None):
    """Construct a RAGPipeline with every external collaborator faked."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]

    store = MagicMock()
    store.query.return_value = vector_response if vector_response is not None else {
        "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]],
    }

    llm = MagicMock()
    llm.generate.return_value = llm_answer

    with ExitStack() as stack:
        stack.enter_context(patch("src.business.rag.retrieval.load_dotenv"))
        stack.enter_context(patch("src.business.rag.retrieval.create_embedder", return_value=embedder))
        stack.enter_context(patch("src.business.rag.retrieval.create_vector_store", return_value=store))
        stack.enter_context(patch("src.business.rag.retrieval.CrossEncoderReRanker"))
        stack.enter_context(patch("src.business.rag.retrieval.create_llm", return_value=llm))
        pipeline = RAGPipeline(persist_dir="/tmp/unused", reranker_config=config)

    return pipeline, embedder, store, llm


def chroma_response(n=3):
    return {
        "ids": [[f"chunk-{i}" for i in range(n)]],
        "documents": [[f"text of chunk {i}" for i in range(n)]],
        "metadatas": [[{"source_id": "doc.pdf", "section": "text"} for _ in range(n)]],
        "distances": [[0.1 * i for i in range(n)]],
    }


class TestRetrieve:
    """_retrieve unpacks the vector store's nested-list response."""

    def test_embeds_the_query_once(self):
        pipeline, embedder, _, _ = build_pipeline(chroma_response())
        pipeline._retrieve("what is X?")
        embedder.embed_query.assert_called_once_with("what is X?")

    def test_builds_one_retrieved_chunk_per_hit(self):
        pipeline, _, _, _ = build_pipeline(chroma_response(3))
        chunks = pipeline._retrieve("q")
        assert len(chunks) == 3
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    def test_maps_every_field_from_the_response(self):
        pipeline, _, _, _ = build_pipeline(chroma_response(1))
        chunk = pipeline._retrieve("q")[0]
        assert chunk.chunk_id == "chunk-0"
        assert chunk.text == "text of chunk 0"
        assert chunk.metadata == {"source_id": "doc.pdf", "section": "text"}
        assert chunk.vector_score == 0.0

    def test_passes_top_k_through_to_the_store(self):
        pipeline, _, store, _ = build_pipeline(chroma_response())
        pipeline._retrieve("q", top_k=17)
        assert store.query.call_args.args[1] == 17

    def test_empty_index_returns_empty_list(self):
        pipeline, _, _, _ = build_pipeline()
        assert pipeline._retrieve("q") == []

    def test_missing_keys_do_not_raise(self):
        """A backend returning a partial dict degrades to no chunks."""
        pipeline, _, _, _ = build_pipeline({"ids": [[]]})
        assert pipeline._retrieve("q") == []

    def test_null_distance_becomes_zero(self):
        """Some backends omit distances; float(None) would raise."""
        response = chroma_response(1)
        response["distances"] = [[None]]
        pipeline, _, _, _ = build_pipeline(response)
        assert pipeline._retrieve("q")[0].vector_score == 0.0

    def test_null_metadata_becomes_empty_dict(self):
        response = chroma_response(1)
        response["metadatas"] = [[None]]
        pipeline, _, _, _ = build_pipeline(response)
        assert pipeline._retrieve("q")[0].metadata == {}


class TestAnswer:
    """answer() orchestrates retrieve → select_context → generate."""

    def test_retrieves_using_the_configured_top_k_input(self):
        config = ReRankerConfig(top_k_input=12)
        pipeline, _, store, _ = build_pipeline(chroma_response(), config=config)
        with patch("src.business.rag.retrieval.select_context", return_value=([], "none")):
            pipeline.answer("q")
        assert store.query.call_args.args[1] == 12

    def test_sends_only_reranked_text_to_the_llm(self):
        """The LLM must see the SELECTED context, not raw retrieval order.

        This is the boundary the re-ranker exists to defend — passing the raw
        top-k through would silently undo the precision gate that keeps
        irrelevant chunks out of the prompt.
        """
        pipeline, _, _, llm = build_pipeline(chroma_response(5))
        selected = [
            ReRankedChunk(chunk_id="keep", text="relevant text", metadata={},
                          vector_score=0.9, rerank_score=0.8)
        ]
        with patch("src.business.rag.retrieval.select_context", return_value=(selected, "high")):
            pipeline.answer("q")

        llm.generate.assert_called_once_with("q", ["relevant text"])

    def test_returns_answer_and_selected_chunks(self):
        pipeline, _, _, _ = build_pipeline(chroma_response(), llm_answer="THE ANSWER")
        selected = [ReRankedChunk("id", "t", {}, 0.1, 0.9)]
        with patch("src.business.rag.retrieval.select_context", return_value=(selected, "high")):
            answer, chunks = pipeline.answer("q")

        assert answer == "THE ANSWER"
        assert chunks == selected

    def test_empty_context_still_calls_the_llm_with_no_context(self):
        """fail_closed / empty index path — must not crash before generating."""
        pipeline, _, _, llm = build_pipeline(llm_answer="I don't know.")
        with patch("src.business.rag.retrieval.select_context", return_value=([], "none")):
            answer, chunks = pipeline.answer("q")

        llm.generate.assert_called_once_with("q", [])
        assert chunks == []

    def test_passes_hybrid_policy_to_the_orchestrator(self):
        pipeline, _, _, _ = build_pipeline(chroma_response())
        with patch("src.business.rag.retrieval.select_context",
                   return_value=([], "none")) as mock_select:
            pipeline.answer("q")
        assert mock_select.call_args.kwargs["policy"] == "hybrid"

    def test_query_is_forwarded_verbatim_to_the_reranker(self):
        pipeline, _, _, _ = build_pipeline(chroma_response())
        with patch("src.business.rag.retrieval.select_context",
                   return_value=([], "none")) as mock_select:
            pipeline.answer("  exact question text  ")
        assert mock_select.call_args.kwargs["query"] == "  exact question text  "
