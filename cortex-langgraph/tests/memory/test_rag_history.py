"""Tests for RagHistoryManager — saved PDF questions and answers."""

from __future__ import annotations

import sqlite3

from src.memory.rag_history import RagHistoryManager

SOURCES = [{"text": "chunk", "metadata": {"source_id": "book.pdf"}, "score": 8.1}]


def test_saved_qa_is_listed_with_its_sources(tmp_db_path):
    history = RagHistoryManager(tmp_db_path)
    history.save("1", "What is RAG?", "Retrieval-augmented generation.", SOURCES)
    (item,) = history.list("1")
    assert (item["question"], item["answer"], item["sources"]) == \
        ("What is RAG?", "Retrieval-augmented generation.", SOURCES)
    assert item["created_at"]


def test_list_is_oldest_first(tmp_db_path):
    history = RagHistoryManager(tmp_db_path)
    for q in ("first", "second", "third"):
        history.save("1", q, "a", [])
    assert [i["question"] for i in history.list("1")] == ["first", "second", "third"]


def test_limit_keeps_the_most_recent(tmp_db_path):
    history = RagHistoryManager(tmp_db_path)
    for i in range(5):
        history.save("1", f"q{i}", "a", [])
    assert [i["question"] for i in history.list("1", limit=2)] == ["q3", "q4"]


def test_history_is_per_user(tmp_db_path):
    history = RagHistoryManager(tmp_db_path)
    history.save("1", "mine", "a", [])
    history.save("2", "theirs", "a", [])
    assert [i["question"] for i in history.list("1")] == ["mine"]


def test_clear_removes_only_that_users_history(tmp_db_path):
    history = RagHistoryManager(tmp_db_path)
    history.save("1", "mine", "a", [])
    history.save("2", "theirs", "a", [])
    assert history.clear("1") == 1
    assert history.list("1") == [] and len(history.list("2")) == 1


def test_survives_a_new_instance(tmp_db_path):
    """The point of the table: Q&A outlives a mode switch, reload or restart."""
    RagHistoryManager(tmp_db_path).save("1", "q", "a", [])
    assert len(RagHistoryManager(tmp_db_path).list("1")) == 1


def test_shares_the_chat_database_without_touching_chat_tables(tmp_db_path):
    from src.memory.chat_history_manager import ChatHistoryManager

    chats = ChatHistoryManager(tmp_db_path)
    chats.ensure_session("s1", "1", "chat")
    RagHistoryManager(tmp_db_path).save("1", "q", "a", [])
    assert [s["id"] for s in chats.list_sessions("1")] == ["s1"]   # Q&A is not a chat session
    tables = {r[0] for r in sqlite3.connect(tmp_db_path).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "messages", "rag_history"} <= tables
