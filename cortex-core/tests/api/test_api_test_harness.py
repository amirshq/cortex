"""Guards the guards in tests/api/conftest.py.

Every API test relies on autouse fixtures to keep it hermetic and away from
real data. If an edit to conftest.py silently disabled one of them, all other
tests would still pass — while quietly reaching the network or the real
data/ directory. These tests fail loudly instead.
"""

from __future__ import annotations

import os
import socket

import pytest


def test_outbound_connections_are_blocked():
    with pytest.raises(RuntimeError, match="network access blocked"):
        socket.create_connection(("93.184.216.34", 443), timeout=1)


def test_dns_lookups_of_external_hosts_are_blocked():
    with pytest.raises(RuntimeError, match="network access blocked"):
        socket.getaddrinfo("api.openai.com", 443)


def test_real_credentials_are_replaced_by_dummies():
    for name in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_SEARCH_API_KEY", "NEWS_API_KEY"):
        assert os.environ[name].startswith("test-dummy-"), name


def test_live_data_and_storage_are_pinned_to_safe_values(tmp_path):
    assert os.environ["LIVE_DATA_PROVIDER"] == "mock"
    assert os.environ["SQLITE_DB_PATH"] == str(tmp_path / "chatbot.db")


def test_project_writes_are_redirected_into_tmp(sandboxed_paths, tmp_path):
    import src.api.controller as controller
    import src.business.rag as rag

    assert controller._PROJECT_ROOT == sandboxed_paths.project_root
    assert rag._PROJECT_ROOT == sandboxed_paths.project_root
    assert sandboxed_paths.project_root.is_relative_to(tmp_path)
    # ChatHistoryManager stores a resolved Path, not the string it was given.
    assert controller.ChatHistoryManager().db_path == sandboxed_paths.db_path.resolve()


def test_each_test_gets_a_full_rate_limit_bucket():
    import src.api.router as router

    assert router._rate_limiter.tokens == router._rate_limiter.capacity
