"""Fixtures for the end-to-end API tests.

End-to-end here means a real HTTP request runs through the real app AND the
real business wiring behind the controller:

    chat: process_chat_message → _make_chatbot → AgenticChatbot (LangGraph)
          → LongTermMemory on a REAL Chroma store and ChatHistoryManager on a
          real SQLite file (both in tmp_path, via VECTORDB_DIR / SQLITE_DB_PATH)
    RAG:  query_rag / ingest_pdfs, including source shaping and RAG metrics

Only true external boundaries are faked:

    chat: the chat model (FakeChatModel), the embedding API, Redis
    RAG:  RAGPipeline (embedder + Chroma + cross-encoder + LLM), and the
          index reset + index builder behind ingest_pdfs

The RAG boundary sits at RAGPipeline because the real cross-encoder needs
torch, a model download and ~1s per request — incompatible with the hermetic
default tier. Retrieval quality itself is measured in tests/evals/.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import FakeChatModel, FakeEmbedder, FakeRedisMemory, reply, tool_call


@pytest.fixture
def chat_stack(monkeypatch) -> SimpleNamespace:
    """The chat path with its three external services replaced by fakes.

    _make_chatbot builds new collaborators on every request. Returning the SAME
    fake model / Redis each time lets state carry across requests, the way the
    real services would. The conversation store is the real Chroma one, so its
    state carries across requests on disk.
    """
    import os

    import src.business.chatbot as chatbot_package
    import src.business.chatbot.agentic_chatbot as agentic
    from src.memory.vectordb import create_conversation_vector_store

    embedder = FakeEmbedder()
    redis = FakeRedisMemory()
    llm = FakeChatModel()

    monkeypatch.setattr(chatbot_package, "create_embedder", lambda **kwargs: embedder)
    monkeypatch.setattr(chatbot_package, "create_memory", lambda **kwargs: redis)
    monkeypatch.setattr(agentic, "create_llm", lambda *args, **kwargs: llm)
    # AgenticChatbot.__init__ calls load_dotenv() per request; keep .env out entirely.
    monkeypatch.setattr(agentic, "load_dotenv", lambda *args, **kwargs: None)

    def queue_replies(*messages) -> None:
        """Script what the model returns, in order, across all requests."""
        llm.responses.extend(messages)

    def memory_rows():
        """Every row in the long-term-memory store, read straight from Chroma."""
        store = create_conversation_vector_store(
            collection_name="chat_history",
            embedding=embedder,
            persist_directory=os.environ["VECTORDB_DIR"],
        )
        got = store.get()
        return [{"text": d, "metadata": m} for d, m in zip(got["documents"], got["metadatas"])]

    return SimpleNamespace(
        llm=llm,
        redis=redis,
        embedder=embedder,
        memory_rows=memory_rows,
        queue_replies=queue_replies,
        reply=reply,
        tool_call=lambda name, args, call_id="call-1": tool_call(name, args, call_id),
    )


@pytest.fixture
def rag_pipeline(monkeypatch) -> SimpleNamespace:
    """Replaces RAGPipeline inside query_rag. Configure .answer/.chunks/.error."""
    import src.business.rag as rag

    state = SimpleNamespace(answer="", chunks=[], confidence="high", error=None,
                            constructed_with=[], questions=[])

    class _Pipeline:
        def __init__(self, persist_dir, **kwargs):
            state.constructed_with.append(persist_dir)

        def answer(self, question):
            state.questions.append(question)
            if state.error:
                raise state.error
            return state.answer, list(state.chunks), state.confidence

    monkeypatch.setattr(rag, "RAGPipeline", _Pipeline)
    return state


@pytest.fixture
def rag_index(monkeypatch) -> SimpleNamespace:
    """Replaces the index reset and index builder behind ingest_pdfs.

    Starts with one chunk "from a previous upload", so tests can see whether
    an upload wiped, replaced or preserved existing index content.
    """
    import src.business.rag as rag

    state = SimpleNamespace(
        chunks=["chunk from a previous upload"],
        events=[],
        build_calls=[],
        build_error=None,
        result=(1, 7),
    )

    def fake_reset_vector_store(persist_dir, **kwargs):
        state.events.append("reset")
        state.chunks.clear()

    def fake_build_index(data_dir, persist_dir, **kwargs):
        state.events.append("build")
        state.build_calls.append({
            "data_dir": Path(data_dir),
            "persist_dir": Path(persist_dir),
            "pdfs": sorted(p.name for p in Path(data_dir).glob("*.pdf")),
        })
        if state.build_error:
            raise state.build_error
        docs, chunks = state.result
        state.chunks.extend(f"chunk {i}" for i in range(chunks))
        return docs, chunks

    monkeypatch.setattr(rag, "reset_vector_store", fake_reset_vector_store)
    monkeypatch.setattr(rag, "build_index", fake_build_index)
    return state
