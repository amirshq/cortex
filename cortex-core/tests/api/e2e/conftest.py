"""Fixtures for the end-to-end API tests.

End-to-end here means a real HTTP request runs through the real app AND the
real business wiring behind the controller:

    chat: process_chat_message → _make_chatbot → AgenticChatbot → LongTermMemory
          → ChatHistoryManager on a real SQLite file (in tmp_path)
    RAG:  query_rag / ingest_pdfs, including source shaping and RAG metrics

Only true external boundaries are faked:

    chat: the OpenAI client, the embedder, Redis, the Chroma conversation store
    RAG:  RAGPipeline (embedder + Chroma + cross-encoder + LLM), and the vector
          store + index builder behind ingest_pdfs

The RAG boundary sits at RAGPipeline because the real cross-encoder needs
torch, a model download and ~1s per request — incompatible with the hermetic
default tier. Retrieval quality itself is measured in tests/evals/.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import (
    FakeConversationVectorStore,
    FakeEmbedder,
    FakeMessage,
    FakeOpenAIClient,
    FakeRedisMemory,
    FakeToolCall,
)


@pytest.fixture
def chat_stack(monkeypatch) -> SimpleNamespace:
    """The chat path with its four external services replaced by stateful fakes.

    _make_chatbot builds new collaborators on every request. Returning the SAME
    fake each time lets state carry across requests, the way real Redis and
    Chroma would.
    """
    import src.business.chatbot as chatbot_package
    import src.business.chatbot.agentic_chatbot as agentic

    embedder = FakeEmbedder()
    redis = FakeRedisMemory()
    conversation_store = FakeConversationVectorStore()
    llm = FakeOpenAIClient([])

    monkeypatch.setattr(chatbot_package, "create_embedder", lambda **kwargs: embedder)
    monkeypatch.setattr(chatbot_package, "create_memory", lambda **kwargs: redis)
    monkeypatch.setattr(chatbot_package, "create_conversation_vector_store",
                        lambda **kwargs: conversation_store)
    monkeypatch.setattr(agentic, "OpenAI", lambda **kwargs: llm)
    # AgenticChatbot.__init__ calls load_dotenv() per request; keep .env out entirely.
    monkeypatch.setattr(agentic, "load_dotenv", lambda *args, **kwargs: None)

    def queue_replies(*messages: FakeMessage) -> None:
        """Script what the model returns, in order, across all requests."""
        llm.completions._scripted.extend(messages)

    return SimpleNamespace(
        llm=llm,
        redis=redis,
        conversation_store=conversation_store,
        embedder=embedder,
        queue_replies=queue_replies,
        reply=lambda text: FakeMessage(content=text),
        tool_call=lambda name, arguments, call_id="call-1":
            FakeMessage(tool_calls=[FakeToolCall(call_id, name, arguments)]),
    )


@pytest.fixture
def rag_pipeline(monkeypatch) -> SimpleNamespace:
    """Replaces RAGPipeline inside query_rag. Configure .answer/.chunks/.error."""
    import src.business.rag as rag

    state = SimpleNamespace(answer="", chunks=[], error=None, constructed_with=[], questions=[])

    class _Pipeline:
        def __init__(self, persist_dir, **kwargs):
            state.constructed_with.append(persist_dir)

        def answer(self, question):
            state.questions.append(question)
            if state.error:
                raise state.error
            return state.answer, list(state.chunks)

    monkeypatch.setattr(rag, "RAGPipeline", _Pipeline)
    return state


@pytest.fixture
def rag_index(monkeypatch) -> SimpleNamespace:
    """Replaces the vector store and index builder behind ingest_pdfs.

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

    class _Store:
        def reset(self):
            state.events.append("reset")
            state.chunks.clear()

    def fake_create_vector_store(persist_dir, **kwargs):
        state.events.append("open_store")
        return _Store()

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

    monkeypatch.setattr(rag, "create_vector_store", fake_create_vector_store)
    monkeypatch.setattr(rag, "build_index", fake_build_index)
    return state
