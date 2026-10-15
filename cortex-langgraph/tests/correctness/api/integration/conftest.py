"""Business-layer stand-ins for the HTTP integration tests.

Integration tests run the real app, middleware, router, rate-limit dependency
and controllers. Only the business entry points below — and ChatHistoryManager
for the session routes — are replaced, so a test can assert on exactly what
the HTTP layer handed to the business layer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def chat_business(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value={"reply": "Hi there", "model_used": "gpt-4o", "tokens_used": 12})
    monkeypatch.setattr("src.api.controller.process_chat_message", mock)
    return mock


@pytest.fixture
def history_business(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value={
        "messages": [{"role": "user", "content": "Hello", "timestamp": "2026-09-16T10:00:00"}],
        "total": 1,
        "session_id": "s1",
    })
    monkeypatch.setattr("src.api.controller.get_chat_history", mock)
    return mock


@pytest.fixture
def session_store(monkeypatch) -> MagicMock:
    """The ChatHistoryManager instance the session routes will receive."""
    store = MagicMock()
    store.list_sessions.return_value = []
    store.delete_session.return_value = True
    monkeypatch.setattr("src.api.controller.ChatHistoryManager", lambda *args, **kwargs: store)
    return store


@pytest.fixture
def rag_query_business(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value={
        "answer": "The warranty is 24 months.",
        "sources": [{"text": "Warranty: 24 months.", "metadata": {"source_id": "manual.pdf"}, "score": 0.91}],
    })
    monkeypatch.setattr("src.api.controller.query_rag", mock)
    return mock


@pytest.fixture
def ingest_business(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value={"docs_indexed": 1, "chunks_indexed": 12, "table_ocr_enabled": True})
    monkeypatch.setattr("src.api.controller.ingest_pdfs", mock)
    return mock
