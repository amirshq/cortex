"""Fixtures shared by every test under tests/api/.

Three jobs, all autouse so that no test can forget them:

1. Hermetic environment.
   src.* calls load_dotenv() at import time AND inside AgenticChatbot.__init__
   on every chat request. load_dotenv() never overrides a variable that is
   already set — so deleting a credential is useless (the next load_dotenv()
   puts the real one straight back from .env). Instead every credential is SET
   to a dummy value, and a socket guard turns any real outbound connection
   into a loud test failure.

2. Data safety.
   The API writes to real project paths: RAGController.upload deletes every
   PDF in <project>/data/rag_uploads, ingest_pdfs resets the RAG index under
   <project>/data/rag_vectorstore, and the session endpoints open the relative
   path data/chatbot.db. Every one of those is redirected into tmp_path.

3. Shared state.
   src.api.router._rate_limiter is one module-level TokenBucket shared by all
   requests. A fresh bucket per test keeps tests independent of run order.
"""

from __future__ import annotations

import ipaddress
import socket
from functools import partial
from types import SimpleNamespace
from typing import Callable, Dict, Optional

import pytest


# ---------------------------------------------------------------------------
# 1. Hermetic environment
# ---------------------------------------------------------------------------
_CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "AZURE_OPENAI_API_KEY",
    "AZURE_SEARCH_API_KEY",
    "NEWS_API_KEY",
    "UNSTRUCTURED_API_KEY",
)
_ENDPOINT_VARS = {
    "AZURE_OPENAI_ENDPOINT": "https://example.invalid/openai",
    "AZURE_SEARCH_ENDPOINT": "https://example.invalid/search",
    "AZURE_REDIS_CONNECTION_STRING": "rediss://:dummy@example.invalid:6380/0",
    # Port 1 on loopback: nothing listens there, so a missed Redis mock fails fast.
    "REDIS_URL": "redis://127.0.0.1:1/0",
}
# Local, non-cloud providers. LIVE_DATA_PROVIDER=mock means web_search never
# reaches DuckDuckGo/NewsAPI even if a test lets the real tool run.
_SAFE_PROVIDERS = {
    "LLM_PROVIDER": "openai",
    "EMBEDDING_PROVIDER": "openai",
    "VECTOR_STORE_PROVIDER": "chroma",
    "CHAT_VECTOR_STORE_PROVIDER": "chroma",
    "MEMORY_PROVIDER": "redis",
    "LIVE_DATA_PROVIDER": "mock",
}


def _is_loopback(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    if host in (None, "", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch, tmp_path):
    """Dummy credentials, local providers, and storage paths inside tmp_path."""
    for name in _CREDENTIAL_VARS:
        monkeypatch.setenv(name, f"test-dummy-{name.lower()}")
    for name, value in {**_ENDPOINT_VARS, **_SAFE_PROVIDERS}.items():
        monkeypatch.setenv(name, value)
    # The chat path and GET /history resolve these; absolute tmp paths mean a
    # test that reaches real storage code still never touches data/.
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "chatbot.db"))
    monkeypatch.setenv("VECTORDB_DIR", str(tmp_path / "vectordb"))


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail any test that tries to open a non-loopback connection.

    TestClient drives the ASGI app in-process and opens no sockets, so this
    costs the HTTP tests nothing — it only fires when a mock was missed and
    real client code (OpenAI, Redis, a live-data provider) tries to go out.
    """
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock, address):
        if isinstance(address, tuple) and not _is_loopback(address[0]):
            raise RuntimeError(f"network access blocked in API tests: connect to {address!r}")
        return real_connect(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_loopback(host):
            raise RuntimeError(f"network access blocked in API tests: DNS lookup of {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


# ---------------------------------------------------------------------------
# 2. Data safety
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def sandboxed_paths(monkeypatch, tmp_path) -> SimpleNamespace:
    """Point every project-root-relative write at tmp_path.

    Tests that need to look at what was written (uploaded PDFs, the SQLite
    file) request this fixture by name to get the paths back.
    """
    import src.api.controller as controller
    import src.business.chatbot as chatbot
    import src.business.rag as rag
    from src.memory.chat_history_manager import ChatHistoryManager

    project_root = tmp_path / "project"
    uploads_dir = project_root / "data" / "rag_uploads"
    # Created up front because the older upload tests in test_controller.py
    # patch Path.mkdir but not dest.open("wb"). Before this fixture existed
    # those tests opened — and truncated — the REAL data/rag_uploads/test.pdf
    # on every run; with the root redirected, the directory must already exist.
    uploads_dir.mkdir(parents=True)
    db_path = tmp_path / "chatbot.db"

    monkeypatch.setattr(controller, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(rag, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(chatbot, "_PROJECT_ROOT", project_root)
    # list_sessions/delete_session call ChatHistoryManager() with its relative
    # default "data/chatbot.db". Bind them to the same tmp database the chat
    # path uses via SQLITE_DB_PATH, so one workflow sees one database.
    monkeypatch.setattr(controller, "ChatHistoryManager",
                        partial(ChatHistoryManager, db_path=str(db_path)))

    return SimpleNamespace(
        project_root=project_root,
        uploads_dir=uploads_dir,
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# 3. Shared state
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def fresh_rate_limiter(monkeypatch):
    """Replace the module-level bucket with a fresh one, same production config.

    Capacity and refill rate are copied from the real bucket rather than
    hard-coded, so these tests follow the production settings if they change.
    """
    import src.api.router as router
    from src.api.ratelimiter import TokenBucket

    production = router._rate_limiter
    bucket = TokenBucket(capacity=production.capacity, refill_rate=production.refill_rate)
    monkeypatch.setattr(router, "_rate_limiter", bucket)
    return bucket


class FakeClock:
    """Stands in for the `time` module inside src.api.ratelimiter only."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch) -> FakeClock:
    """Controllable time for the rate limiter — no real sleeps.

    Patches the `time` name inside src.api.ratelimiter, never the global
    time.time (which every other library in the process also uses). The
    bucket is rebuilt AFTER patching so its last-refill timestamp comes from
    the fake clock too; a bucket created on the real clock would compute a
    huge negative elapsed time on its first refill.
    """
    import src.api.ratelimiter as ratelimiter
    import src.api.router as router

    clock = FakeClock()
    monkeypatch.setattr(ratelimiter, "time", clock)
    production = router._rate_limiter
    monkeypatch.setattr(router, "_rate_limiter",
                        ratelimiter.TokenBucket(capacity=production.capacity,
                                                refill_rate=production.refill_rate))
    return clock


# ---------------------------------------------------------------------------
# HTTP client and metrics
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    """TestClient over the real app: middleware, router, dependencies, controllers.

    raise_server_exceptions=False so an unhandled error comes back as the 500
    response a real client would see, instead of being re-raised in the test.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def metric() -> Callable[..., float]:
    """Read a Prometheus sample from the process-global registry.

    The registry is shared by every test in the run, so assert on the change
    across a request (after - before), never on an absolute value.
    """
    from prometheus_client import REGISTRY

    def read(name: str, labels: Optional[Dict[str, str]] = None) -> float:
        return REGISTRY.get_sample_value(name, labels or {}) or 0.0

    return read
