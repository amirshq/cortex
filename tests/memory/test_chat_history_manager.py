"""Tests for ChatHistoryManager — the durable SQLite transcript.

These run against a real SQLite file in tmp_path (SQLite needs no server,
so there is no reason to mock it — and mocking would test nothing, since
the whole class is SQL).

Includes delete_session(), which shipped in the session-management commit
without any test.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.memory.chat_history_manager import ChatHistoryManager


@pytest.fixture
def manager(tmp_db_path) -> ChatHistoryManager:
    return ChatHistoryManager(db_path=tmp_db_path)


class TestSchemaSetup:
    def test_creates_the_database_file(self, tmp_db_path):
        ChatHistoryManager(db_path=tmp_db_path)
        import os
        assert os.path.exists(tmp_db_path)

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "chat.db"
        ChatHistoryManager(db_path=str(nested))
        assert nested.exists()

    def test_creates_both_tables(self, manager):
        with sqlite3.connect(manager.db_path) as conn:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"sessions", "messages"} <= names

    def test_init_is_idempotent(self, tmp_db_path):
        """The constructor runs on every request — it must not wipe data."""
        first = ChatHistoryManager(db_path=tmp_db_path)
        first.ensure_session("s1", "u1")
        first.save_message("s1", "u1", "user", "kept")

        second = ChatHistoryManager(db_path=tmp_db_path)
        assert len(second.get_messages("s1")) == 1


class TestSessions:
    def test_ensure_session_creates_a_row(self, manager):
        manager.ensure_session("s1", "u1", "My chat")
        assert manager.list_sessions("u1")[0]["title"] == "My chat"

    def test_ensure_session_is_idempotent(self, manager):
        manager.ensure_session("s1", "u1", "First title")
        manager.ensure_session("s1", "u1", "Second title")

        sessions = manager.list_sessions("u1")
        assert len(sessions) == 1
        assert sessions[0]["title"] == "First title", "re-ensuring must not retitle"

    def test_default_title(self, manager):
        manager.ensure_session("s1", "u1")
        assert manager.list_sessions("u1")[0]["title"] == "New conversation"

    def test_update_session_title(self, manager):
        manager.ensure_session("s1", "u1", "old")
        manager.update_session_title("s1", "new")
        assert manager.list_sessions("u1")[0]["title"] == "new"

    def test_list_sessions_is_scoped_to_the_user(self, manager):
        manager.ensure_session("s1", "u1")
        manager.ensure_session("s2", "u2")
        assert [s["id"] for s in manager.list_sessions("u1")] == ["s1"]

    def test_list_sessions_is_newest_first(self, manager):
        import time
        manager.ensure_session("older", "u1")
        time.sleep(0.01)
        manager.ensure_session("newer", "u1")
        assert [s["id"] for s in manager.list_sessions("u1")] == ["newer", "older"]

    def test_list_sessions_for_unknown_user_is_empty(self, manager):
        assert manager.list_sessions("nobody") == []

    def test_list_sessions_returns_the_documented_keys(self, manager):
        manager.ensure_session("s1", "u1")
        assert set(manager.list_sessions("u1")[0]) == {"id", "title", "created_at"}


class TestDeleteSession:
    """Shipped untested in the session-management commit."""

    def test_deletes_the_session(self, manager):
        manager.ensure_session("s1", "u1")
        assert manager.delete_session("s1", "u1") is True
        assert manager.list_sessions("u1") == []

    def test_deletes_the_sessions_messages_too(self, manager):
        """Otherwise the rows are orphaned and count_messages still sees them."""
        manager.ensure_session("s1", "u1")
        manager.save_message("s1", "u1", "user", "hello")
        manager.save_message("s1", "u1", "assistant", "hi")

        manager.delete_session("s1", "u1")
        assert manager.get_messages("s1") == []
        assert manager.count_messages("s1") == 0

    def test_returns_false_for_an_unknown_session(self, manager):
        assert manager.delete_session("nope", "u1") is False

    def test_returns_false_when_the_session_belongs_to_another_user(self, manager):
        """Ownership check — u2 must not be able to delete u1's session."""
        manager.ensure_session("s1", "u1")
        assert manager.delete_session("s1", "u2") is False
        assert len(manager.list_sessions("u1")) == 1

    def test_leaves_other_sessions_alone(self, manager):
        manager.ensure_session("s1", "u1")
        manager.ensure_session("s2", "u1")
        manager.save_message("s2", "u1", "user", "survivor")

        manager.delete_session("s1", "u1")
        assert [s["id"] for s in manager.list_sessions("u1")] == ["s2"]
        assert len(manager.get_messages("s2")) == 1

    def test_wrong_user_still_deletes_the_messages(self, manager):
        """KNOWN DEFECT, pinned deliberately.

        delete_session() deletes from `messages` before checking session
        ownership, so a mismatched user_id returns False (correct) but has
        already wiped the messages (wrong) — the session row survives with
        an empty transcript.

        Fix: move the message DELETE after the session DELETE and make it
        conditional on rowcount, or wrap both in one ownership-scoped
        transaction. When that lands, this test should be rewritten to
        assert the messages SURVIVE.
        """
        manager.ensure_session("s1", "u1")
        manager.save_message("s1", "u1", "user", "should have survived")

        assert manager.delete_session("s1", "attacker") is False
        assert len(manager.list_sessions("u1")) == 1
        assert manager.get_messages("s1") == []


class TestMessages:
    def test_save_and_read_back(self, manager):
        manager.ensure_session("s1", "u1")
        manager.save_message("s1", "u1", "user", "hello")

        messages = manager.get_messages("s1")
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"
        assert messages[0]["timestamp"]

    def test_messages_come_back_oldest_first(self, manager):
        manager.ensure_session("s1", "u1")
        for i in range(5):
            manager.save_message("s1", "u1", "user", f"m{i}")
        assert [m["content"] for m in manager.get_messages("s1")] == [f"m{i}" for i in range(5)]

    def test_limit_truncates(self, manager):
        manager.ensure_session("s1", "u1")
        for i in range(10):
            manager.save_message("s1", "u1", "user", f"m{i}")
        assert len(manager.get_messages("s1", limit=3)) == 3

    def test_offset_paginates(self, manager):
        manager.ensure_session("s1", "u1")
        for i in range(10):
            manager.save_message("s1", "u1", "user", f"m{i}")
        page = manager.get_messages("s1", limit=3, offset=3)
        assert [m["content"] for m in page] == ["m3", "m4", "m5"]

    def test_messages_are_scoped_to_the_session(self, manager):
        manager.ensure_session("s1", "u1")
        manager.ensure_session("s2", "u1")
        manager.save_message("s1", "u1", "user", "in s1")
        manager.save_message("s2", "u1", "user", "in s2")
        assert [m["content"] for m in manager.get_messages("s1")] == ["in s1"]

    def test_unknown_session_returns_empty(self, manager):
        assert manager.get_messages("nope") == []

    def test_count_messages(self, manager):
        manager.ensure_session("s1", "u1")
        for i in range(7):
            manager.save_message("s1", "u1", "user", f"m{i}")
        assert manager.count_messages("s1") == 7

    def test_count_messages_for_unknown_session_is_zero(self, manager):
        assert manager.count_messages("nope") == 0

    def test_unicode_and_newlines_round_trip(self, manager):
        manager.ensure_session("s1", "u1")
        content = "سلام — emoji 🎉\nsecond line\ttab"
        manager.save_message("s1", "u1", "user", content)
        assert manager.get_messages("s1")[0]["content"] == content

    def test_sql_metacharacters_are_parameterised_not_interpolated(self, manager):
        """Content is bound, never formatted into the statement."""
        manager.ensure_session("s1", "u1")
        payload = "'); DROP TABLE messages; --"
        manager.save_message("s1", "u1", "user", payload)

        assert manager.get_messages("s1")[0]["content"] == payload
        assert manager.count_messages("s1") == 1


class TestGetLatestSummary:
    """Cold-start hydration: pull the previous session forward into a new one."""

    def test_returns_none_when_the_user_has_no_sessions(self, manager):
        assert manager.get_latest_summary("nobody") is None

    def test_returns_none_when_the_session_has_no_messages(self, manager):
        manager.ensure_session("s1", "u1")
        assert manager.get_latest_summary("u1") is None

    def test_formats_role_prefixed_lines(self, manager):
        manager.ensure_session("s1", "u1")
        manager.save_message("s1", "u1", "user", "hello")
        manager.save_message("s1", "u1", "assistant", "hi there")

        assert manager.get_latest_summary("u1") == "User: hello\nAssistant: hi there"

    def test_reads_oldest_to_newest(self, manager):
        """The SQL selects DESC then reverses — a regression here would feed
        the model the conversation backwards."""
        manager.ensure_session("s1", "u1")
        for i in range(4):
            manager.save_message("s1", "u1", "user", f"m{i}")

        lines = manager.get_latest_summary("u1").split("\n")
        assert lines == [f"User: m{i}" for i in range(4)]

    def test_takes_only_the_last_n_messages(self, manager):
        manager.ensure_session("s1", "u1")
        for i in range(20):
            manager.save_message("s1", "u1", "user", f"m{i}")

        lines = manager.get_latest_summary("u1", n_messages=3).split("\n")
        assert lines == ["User: m17", "User: m18", "User: m19"]

    def test_uses_the_most_recent_session_only(self, manager):
        import time
        manager.ensure_session("old", "u1")
        manager.save_message("old", "u1", "user", "stale")
        time.sleep(0.01)
        manager.ensure_session("new", "u1")
        manager.save_message("new", "u1", "user", "fresh")

        summary = manager.get_latest_summary("u1")
        assert "fresh" in summary and "stale" not in summary

    def test_is_scoped_to_the_user(self, manager):
        manager.ensure_session("s1", "u1")
        manager.save_message("s1", "u1", "user", "u1 content")
        assert manager.get_latest_summary("u2") is None
