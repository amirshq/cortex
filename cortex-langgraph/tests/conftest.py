"""Shared pytest configuration and fixtures.

Centralises the sys.path bootstrap and provides the fake collaborators the
orchestration-layer tests need. In cortex-langgraph every fake implements the
real LangChain interface, so production code can't tell it apart from the
real thing:

- FakeEmbedder       — a langchain_core `Embeddings`
- FakeChatModel      — a langchain_core `BaseChatModel` that replays scripted
                       AIMessages (including tool calls) and records every call
- FakeRedisMemory    — the async short-term memory interface (no Redis)
- conversation_store — a REAL Chroma vector store, in tmp_path, embedding with
                       FakeEmbedder. Chroma runs in-process, so it's hermetic.

Nothing here talks to a real service. Tests that DO need real services
live in tests/evals/ and are marked `eval` (deselected by default — see
pytest.ini).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class FakeEmbedder(Embeddings):
    """Deterministic stand-in for OpenAIEmbeddings.

    Returns a fixed-length vector derived from the text so that identical
    text always embeds identically, without any network call. Records every
    call so tests can assert on what was embedded.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.query_calls: List[str] = []
        self.document_calls: List[List[str]] = []

    def _vector(self, text: str) -> List[float]:
        seed = sum(ord(c) for c in text)
        return [float((seed + i) % 97) / 97.0 + 0.01 for i in range(self.dim)]

    def embed_query(self, text: str) -> List[float]:
        self.query_calls.append(text)
        return self._vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.document_calls.append(list(texts))
        return [self._vector(t) for t in texts]


# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------
class FakeChatModel(BaseChatModel):
    """Replays a scripted list of AIMessages, one per model call.

    - `calls` records the full message list of every call, so tests can
      assert on exactly what the graph sent to the model.
    - `bound_tools` records what bind_tools() was given (the agent graph
      binds its tools once, at build time).
    - Running out of scripted replies is a test failure, not a hang — the
      graph called the model more times than the test expected.
    - A scripted exception is raised instead of returned (provider failure).
    """

    responses: List[Any] = Field(default_factory=list)
    calls: List[List[BaseMessage]] = Field(default_factory=list)
    bound_tools: List[Any] = Field(default_factory=list)
    model_name: str = "fake-model"

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError(
                "FakeChatModel ran out of scripted responses — the graph called "
                "the model more times than the test expected."
            )
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response)])


def reply(text: str, usage: Optional[Dict[str, int]] = None) -> AIMessage:
    """A plain-text model answer (ends the agent loop)."""
    return AIMessage(content=text, usage_metadata=_usage(usage))


def tool_call(name: str, args: Dict, call_id: str = "call-1",
              usage: Optional[Dict[str, int]] = None) -> AIMessage:
    """A model turn that asks for one tool call."""
    return tool_calls([(name, args, call_id)], usage=usage)


def tool_calls(calls, usage: Optional[Dict[str, int]] = None) -> AIMessage:
    """A model turn that asks for several tool calls at once: [(name, args, id), ...]."""
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"} for n, a, i in calls],
        usage_metadata=_usage(usage),
    )


def _usage(usage: Optional[Dict[str, int]]):
    if not usage:
        return None
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }


# ---------------------------------------------------------------------------
# Short-term memory
# ---------------------------------------------------------------------------
class FakeRedisMemory:
    """In-memory ShortTermMemoryBase implementation (async, no Redis)."""

    def __init__(self, preload: List[Dict] | None = None):
        self.store: Dict[str, List[Dict]] = {}
        self._preload = preload or []

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        self.store.setdefault(session_id, []).append({"role": role, "content": content})

    async def get_messages(self, session_id: str, limit: int = 10) -> List[Dict]:
        if session_id not in self.store and self._preload:
            return list(self._preload)[-limit:]
        return self.store.get(session_id, [])[-limit:]

    async def clear(self, session_id: str) -> None:
        self.store.pop(session_id, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_network_outside_evals(request, monkeypatch):
    """Fail any non-eval test that opens a non-loopback connection.

    A NewsAPI unit test once silently ran a real web search (search() had
    started delegating to ddgs, and the test only mocked `requests`). Evals
    need the network on purpose, so they are exempt.
    """
    if request.node.get_closest_marker("eval"):
        return
    import ipaddress
    import socket

    def is_loopback(host) -> bool:
        if isinstance(host, bytes):
            host = host.decode()
        if host in (None, "", "localhost"):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    real_connect, real_getaddrinfo = socket.socket.connect, socket.getaddrinfo

    def guarded_connect(sock, address):
        if isinstance(address, tuple) and not is_loopback(address[0]):
            raise RuntimeError(f"network access blocked in unit tests: {address!r}")
        return real_connect(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not is_loopback(host):
            raise RuntimeError(f"network access blocked in unit tests: DNS lookup of {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def conversation_store(tmp_path, fake_embedder):
    """A real (in-process) Chroma conversation store in tmp_path."""
    from src.memory.vectordb import create_conversation_vector_store

    return create_conversation_vector_store(
        collection_name="chat_history",
        embedding=fake_embedder,
        persist_directory=str(tmp_path / "conversation_store"),
        provider="chroma",
    )


@pytest.fixture
def override_config(monkeypatch, tmp_path):
    """Run a test against a modified copy of the real config.yml.

        override_config({"models": {"chat": {"name": "gpt-x"}}})

    Sections are merged one level deep into the real file, so everything not
    mentioned keeps its production value.
    """
    import yaml

    import src.utils.config as config_module

    def apply(overrides: Dict[str, Any]) -> None:
        config = config_module.load_config()
        for section, values in overrides.items():
            if isinstance(values, dict) and isinstance(config.get(section), dict):
                config[section] = {**config[section], **values}
            else:
                config[section] = values
        path = tmp_path / "config.yml"
        path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(config_module, "CONFIG_PATH", path)

    return apply


@pytest.fixture
def fake_redis_memory() -> FakeRedisMemory:
    return FakeRedisMemory()


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_chatbot.db")
