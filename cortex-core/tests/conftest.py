"""Shared pytest configuration and fixtures.

Centralises the sys.path bootstrap that every existing test file does by
hand, and provides the fake collaborators the orchestration-layer tests
need (fake embedder, fake vector store, fake OpenAI client).

Nothing here talks to a real service. Tests that DO need real services
live in tests/evals/ and are marked `eval` (deselected by default — see
pytest.ini).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------
class FakeEmbedder:
    """Deterministic stand-in for OpenAIEmbedder.

    Returns a fixed-length vector derived from the text so that identical
    text always embeds identically (which is what the caching / id-stability
    assertions rely on), without any network call.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.embed_calls: List[str] = []
        self.embed_documents_calls: List[List[str]] = []

    def _vector(self, text: str) -> List[float]:
        seed = sum(ord(c) for c in text)
        return [float((seed + i) % 97) / 97.0 for i in range(self.dim)]

    def embed(self, text: str) -> List[float]:
        self.embed_calls.append(text)
        return self._vector(text)

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embed_documents_calls.append(list(texts))
        return [self._vector(t) for t in texts]


class FakeConversationVectorStore:
    """In-memory ConversationVectorStoreBase implementation."""

    def __init__(self, search_results: List[Dict] | None = None):
        self.rows: List[Dict] = []
        self.deleted_filters: List[Dict] = []
        self._search_results = search_results

    def add(self, ids, embeddings, documents, metadatas) -> None:
        for i, doc, meta in zip(ids, documents, metadatas):
            self.rows.append({"id": i, "text": doc, "metadata": meta})

    def search(self, embedding, top_k: int = 5, filters=None) -> List[Dict]:
        if self._search_results is not None:
            return self._search_results[:top_k]
        rows = self.rows
        if filters and "user_id" in filters:
            rows = [r for r in rows if r["metadata"].get("user_id") == filters["user_id"]]
        return [
            {"text": r["text"], "metadata": r["metadata"], "score": 0.1}
            for r in rows[:top_k]
        ]

    def delete(self, filters: Dict) -> None:
        self.deleted_filters.append(filters)
        user_id = filters.get("user_id")
        self.rows = [r for r in self.rows if r["metadata"].get("user_id") != user_id]


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
# OpenAI chat-completions fakes (for the agent tool-calling loop)
# ---------------------------------------------------------------------------
class FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: List[FakeToolCall] | None = None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


class FakeCompletions:
    """Replays a scripted list of assistant messages, one per create() call.

    Records every `messages` list it was handed so tests can assert on what
    the agent actually sent to the model (system prompt contents, tool
    result plumbing, message ordering).
    """

    def __init__(self, scripted: List[FakeMessage]):
        self._scripted = list(scripted)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._scripted:
            raise AssertionError(
                "FakeCompletions ran out of scripted responses — the agent "
                "loop called the model more times than the test expected."
            )
        message = self._scripted.pop(0)
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class FakeOpenAIClient:
    def __init__(self, scripted: List[FakeMessage]):
        self.completions = FakeCompletions(scripted)
        self.chat = type("Chat", (), {"completions": self.completions})()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_conversation_store() -> FakeConversationVectorStore:
    return FakeConversationVectorStore()


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
