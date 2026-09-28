"""Tests for the ingestion path: Chunker, table-section tagging, build_index.

Docling itself is not exercised — it's a third-party PDF parser and testing
it would test their code, not ours. What IS tested is everything we own on
top of it: the chunk-window arithmetic, chunk-id stability, the provenance
tag that marks which chunks came from tables, and the batching loop that
keeps large PDFs from OOMing the embedder.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.business.rag.index_builder import _chunk_document, build_index
from src.business.rag.pdfingest.chunk import Chunk, Chunker
from src.business.rag.pdfingest.pdf_digest import IngestedDocument


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
class TestChunkerConfiguration:
    def test_overlap_must_be_smaller_than_chunk_size(self):
        """Otherwise `start = end - overlap` never advances and split()
        loops forever building infinite chunks."""
        with pytest.raises(AssertionError, match="overlap must be smaller"):
            Chunker(chunk_size=100, overlap=100)

    def test_defaults(self):
        chunker = Chunker()
        assert chunker.chunk_size == 800
        assert chunker.overlap == 100


class TestChunkerSplit:
    def test_empty_text_returns_no_chunks(self):
        assert Chunker().split("", {"source_id": "d.pdf"}) == []

    def test_whitespace_only_returns_no_chunks(self):
        assert Chunker().split("   \n\t  ", {"source_id": "d.pdf"}) == []

    def test_short_text_becomes_one_chunk(self):
        chunks = Chunker(chunk_size=100, overlap=10).split("short text", {"source_id": "d"})
        assert len(chunks) == 1
        assert chunks[0].text == "short text"

    def test_long_text_is_split(self):
        chunks = Chunker(chunk_size=100, overlap=10).split("x" * 450, {"source_id": "d"})
        assert len(chunks) > 1

    def test_chunks_overlap_by_the_configured_amount(self):
        """Overlap is what stops a fact from being cut in half at a boundary
        and becoming unretrievable."""
        text = "".join(str(i % 10) for i in range(300))
        chunks = Chunker(chunk_size=100, overlap=20).split(text, {"source_id": "d"})

        first_end = chunks[0].metadata["chunk_end"]
        second_start = chunks[1].metadata["chunk_start"]
        assert first_end - second_start == 20

    def test_chunks_cover_the_whole_document(self):
        text = "".join(str(i % 10) for i in range(1000))
        chunks = Chunker(chunk_size=200, overlap=50).split(text, {"source_id": "d"})

        assert chunks[0].metadata["chunk_start"] == 0
        assert chunks[-1].metadata["chunk_end"] == len(text)

    def test_split_terminates_on_text_shorter_than_the_overlap(self):
        """Regression guard for the infinite-loop shape."""
        chunks = Chunker(chunk_size=100, overlap=90).split("x" * 105, {"source_id": "d"})
        assert 0 < len(chunks) < 50

    def test_every_chunk_carries_source_metadata(self):
        chunks = Chunker(chunk_size=100, overlap=10).split("x" * 300, {"source_id": "paper.pdf"})
        assert all(c.metadata["source_id"] == "paper.pdf" for c in chunks)

    def test_every_chunk_records_the_strategy(self):
        """Provenance: which chunking config produced this index."""
        chunks = Chunker(chunk_size=100, overlap=10, strategy_name="v2").split(
            "x" * 300, {"source_id": "d"})
        assert all(c.metadata["chunk_strategy"] == "v2" for c in chunks)

    def test_returns_chunk_dataclasses(self):
        chunks = Chunker().split("text", {"source_id": "d"})
        assert isinstance(chunks[0], Chunk)


class TestChunkIds:
    def test_ids_are_deterministic(self):
        """Re-indexing the same PDF must upsert onto the same ids rather
        than duplicating every chunk."""
        a = Chunker(chunk_size=100, overlap=10).split("x" * 400, {"source_id": "d.pdf"})
        b = Chunker(chunk_size=100, overlap=10).split("x" * 400, {"source_id": "d.pdf"})
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b]

    def test_different_sources_yield_different_ids(self):
        """Otherwise identical boilerplate in two PDFs collides and one
        document's chunk silently overwrites the other's."""
        a = Chunker().split("identical content", {"source_id": "a.pdf"})
        b = Chunker().split("identical content", {"source_id": "b.pdf"})
        assert a[0].chunk_id != b[0].chunk_id

    def test_ids_within_a_document_are_unique(self):
        text = "".join(str(i % 10) for i in range(2000))
        chunks = Chunker(chunk_size=100, overlap=10).split(text, {"source_id": "d.pdf"})
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_id_is_a_sha1_hex_digest(self):
        chunk_id = Chunker().split("text", {"source_id": "d"})[0].chunk_id
        assert len(chunk_id) == 40


# ---------------------------------------------------------------------------
# Table-section tagging
# ---------------------------------------------------------------------------
class TestTableSectionTagging:
    """Chunks after the table marker are tagged section="table".

    This is the only provenance signal distinguishing prose from extracted
    table content, which matters because table chunks read as noise to a
    re-ranker and are worth filtering or boosting differently.
    """

    def _doc(self, text):
        return IngestedDocument(source_id="d.pdf", text_blocks=[], table_text="",
                                combined_text=text, metadata={})

    def test_no_marker_tags_everything_as_text(self):
        chunks = _chunk_document(self._doc("x" * 300), Chunker(chunk_size=100, overlap=10), None)
        assert all(c.metadata["section"] == "text" for c in chunks)

    def test_chunks_before_the_marker_are_text(self):
        chunks = _chunk_document(self._doc("x" * 500), Chunker(chunk_size=100, overlap=10), 300)
        early = [c for c in chunks if c.metadata["chunk_start"] < 300]
        assert early and all(c.metadata["section"] == "text" for c in early)

    def test_chunks_at_or_after_the_marker_are_table(self):
        chunks = _chunk_document(self._doc("x" * 500), Chunker(chunk_size=100, overlap=10), 300)
        late = [c for c in chunks if c.metadata["chunk_start"] >= 300]
        assert late and all(c.metadata["section"] == "table" for c in late)

    def test_marker_at_zero_tags_everything_as_table(self):
        chunks = _chunk_document(self._doc("x" * 300), Chunker(chunk_size=100, overlap=10), 0)
        assert all(c.metadata["section"] == "table" for c in chunks)

    def test_source_id_survives_the_enrichment(self):
        chunks = _chunk_document(self._doc("x" * 300), Chunker(chunk_size=100, overlap=10), None)
        assert all(c.metadata["source_id"] == "d.pdf" for c in chunks)

    def test_enrichment_preserves_chunk_offsets(self):
        chunker = Chunker(chunk_size=100, overlap=10)
        raw = chunker.split("x" * 300, {"source_id": "d.pdf"})
        enriched = _chunk_document(self._doc("x" * 300), chunker, None)
        assert [c.metadata["chunk_start"] for c in enriched] == \
               [c.metadata["chunk_start"] for c in raw]


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------
def make_doc(source_id="d.pdf", text="x" * 5000, table_text=""):
    return IngestedDocument(source_id=source_id, text_blocks=[text], table_text=table_text,
                            combined_text=text, metadata={})


def run_build_index(docs, **kwargs):
    embedder = MagicMock()
    embedder.embed_documents.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
    store = MagicMock()

    with patch("src.business.rag.index_builder.load_dotenv"), \
         patch("src.business.rag.index_builder.ingest_directory", return_value=docs), \
         patch("src.business.rag.index_builder.create_embedder", return_value=embedder), \
         patch("src.business.rag.index_builder.create_vector_store", return_value=store):
        result = build_index(data_dir=Path("/tmp/in"), persist_dir=Path("/tmp/out"), **kwargs)

    return result, embedder, store


class TestBuildIndex:
    def test_returns_document_and_chunk_counts(self):
        (docs_n, chunks_n), _, _ = run_build_index([make_doc(), make_doc("e.pdf")])
        assert docs_n == 2
        assert chunks_n > 0

    def test_no_documents_indexes_nothing(self):
        (docs_n, chunks_n), _, store = run_build_index([])
        assert (docs_n, chunks_n) == (0, 0)
        store.upsert.assert_not_called()

    def test_embeds_in_batches_of_fifty(self):
        """Embedding thousands of chunks in one call OOMs the process and
        can exceed the provider's per-request limit."""
        _, embedder, _ = run_build_index([make_doc(text="x" * 60_000)], chunk_size=100, overlap=10)
        assert all(len(call.args[0]) <= 50 for call in embedder.embed_documents.call_args_list)

    def test_upserts_once_per_batch(self):
        _, embedder, store = run_build_index([make_doc(text="x" * 60_000)],
                                             chunk_size=100, overlap=10)
        assert store.upsert.call_count == embedder.embed_documents.call_count

    def test_connects_to_the_store_lazily(self):
        """No documents → no connection attempt, so an empty upload dir
        doesn't fail on a missing/unreachable vector store."""
        with patch("src.business.rag.index_builder.load_dotenv"), \
             patch("src.business.rag.index_builder.ingest_directory", return_value=[]), \
             patch("src.business.rag.index_builder.create_embedder"), \
             patch("src.business.rag.index_builder.create_vector_store") as create_store:
            build_index(data_dir=Path("/tmp/in"), persist_dir=Path("/tmp/out"))
        create_store.assert_not_called()

    def test_upsert_receives_four_aligned_lists(self):
        _, _, store = run_build_index([make_doc(text="x" * 500)], chunk_size=100, overlap=10)
        kwargs = store.upsert.call_args.kwargs
        n = len(kwargs["ids"])
        assert len(kwargs["embeddings"]) == n
        assert len(kwargs["metadatas"]) == n
        assert len(kwargs["documents"]) == n

    def test_top_k_store_truncates(self):
        (_, chunks_n), _, _ = run_build_index([make_doc(text="x" * 20_000)],
                                              chunk_size=100, overlap=10, top_k_store=5)
        assert chunks_n == 5

    def test_chunk_size_and_overlap_are_forwarded(self):
        (_, small), _, _ = run_build_index([make_doc(text="x" * 4000)], chunk_size=100, overlap=10)
        (_, large), _, _ = run_build_index([make_doc(text="x" * 4000)], chunk_size=1000, overlap=10)
        assert small > large

    def test_multiple_documents_are_all_indexed(self):
        _, _, store = run_build_index(
            [make_doc("a.pdf", "x" * 500), make_doc("b.pdf", "y" * 500)],
            chunk_size=100, overlap=10,
        )
        sources = {m["source_id"] for call in store.upsert.call_args_list
                   for m in call.kwargs["metadatas"]}
        assert sources == {"a.pdf", "b.pdf"}
