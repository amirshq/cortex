"""Tests that run without an API key or a model download.

The OpenAI embedder, the re-ranker model and the LLM are replaced by tiny
fakes; the chunker and the Chroma vector database are the real ones.
Run: pytest
"""

from rag.loader import load_documents, split_into_chunks
from rag.pipeline import RAG
from rag.vector_db import VectorDB


class FakeEmbedder:
    """Counts letters a-z: texts sharing words get similar vectors."""
    def embed(self, texts):
        return [[t.lower().count(c) + 0.01 for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]


class FakeReranker:
    """Scores a chunk by how many question words it contains."""
    def rerank(self, question, chunks, top_n):
        words = set(question.lower().split())
        for chunk in chunks:
            chunk["score"] = len(words & set(chunk["text"].lower().split())) / len(words)
        return sorted(chunks, key=lambda c: c["score"], reverse=True)[:top_n]


class FakeLLM:
    """Records what it was given instead of calling OpenAI."""
    def answer(self, question, chunks):
        self.seen = chunks
        return f"answer from {len(chunks)} chunks"


def make_rag(tmp_path):
    return RAG(embedder=FakeEmbedder(), vector_db=VectorDB(str(tmp_path / "db"), "test"),
               reranker=FakeReranker(), llm=FakeLLM())


def test_chunks_overlap_and_cover_the_text():
    chunks = split_into_chunks("abcdefghijklmnop", chunk_size=10, overlap=3)
    assert chunks == ["abcdefghij", "hijklmnop"]


def test_short_text_is_one_chunk():
    assert split_into_chunks("hello", chunk_size=500, overlap=50) == ["hello"]


def test_loads_the_sample_documents():
    sources = [d["source"] for d in load_documents("data")]
    assert "kestrel-specs.txt" in sources and len(sources) == 5


def test_index_stores_every_chunk(tmp_path):
    rag = make_rag(tmp_path)
    docs, chunks = rag.index("data")
    assert docs == 5 and rag.vector_db.count() == chunks


def test_reindexing_does_not_duplicate(tmp_path):
    rag = make_rag(tmp_path)
    _, chunks = rag.index("data")
    rag.index("data")
    assert rag.vector_db.count() == chunks


def test_ask_sends_only_the_reranked_chunks_to_the_llm(tmp_path):
    rag = make_rag(tmp_path)
    rag.index("data")
    answer, sources = rag.ask("What is the warranty for the Kestrel-7?")

    assert answer == "answer from 3 chunks"
    assert rag.llm.seen == sources and len(sources) == 3
    scores = [c["score"] for c in sources]
    assert scores == sorted(scores, reverse=True)           # best chunk first
    assert "veldrin-support.txt" in [c["source"] for c in sources]
