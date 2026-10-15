"""Tests for LongTermMemory — remember / recall / forget, scoped per user.

These run against a REAL Chroma vector store (in-process, in tmp_path) with a
deterministic fake embedder, so they check the actual round trip through
LangChain's VectorStore interface rather than a hand-written fake of it.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from langchain_text_splitters import CharacterTextSplitter

from src.memory.long_term_memory import LongTermMemory


@pytest.fixture
def memory(conversation_store) -> LongTermMemory:
    return LongTermMemory(vectorstore=conversation_store)


def all_rows(memory: LongTermMemory):
    got = memory.vectorstore.get()
    return [
        {"id": i, "text": d, "metadata": m}
        for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
    ]


class TestRemember:
    def test_writes_one_row_for_short_content(self, memory):
        memory.remember("likes hiking", user_id="u1")
        assert [r["text"] for r in all_rows(memory)] == ["likes hiking"]

    def test_splits_long_content_before_storing(self, conversation_store):
        splitter = CharacterTextSplitter(separator=". ", chunk_size=20, chunk_overlap=0)
        memory = LongTermMemory(conversation_store, text_splitter=splitter)
        memory.remember("First fact here. Second fact here. Third fact here.", user_id="u1")
        assert len(all_rows(memory)) == 3

    def test_embeds_each_chunk(self, conversation_store, fake_embedder):
        splitter = CharacterTextSplitter(separator=". ", chunk_size=20, chunk_overlap=0)
        memory = LongTermMemory(conversation_store, text_splitter=splitter)
        memory.remember("First fact here. Second fact here.", user_id="u1")
        embedded = [t for batch in fake_embedder.document_calls for t in batch]
        assert len(embedded) == 2

    def test_attaches_the_documented_metadata(self, memory):
        memory.remember("fact", user_id="u1", memory_type="preference", importance=3)
        meta = all_rows(memory)[0]["metadata"]
        assert meta["user_id"] == "u1"
        assert meta["type"] == "preference"
        assert meta["importance"] == 3

    def test_defaults_type_and_importance(self, memory):
        memory.remember("fact", user_id="u1")
        meta = all_rows(memory)[0]["metadata"]
        assert meta["type"] == "knowledge"
        assert meta["importance"] == 1

    def test_created_at_is_iso_parseable(self, memory):
        memory.remember("fact", user_id="u1")
        datetime.fromisoformat(all_rows(memory)[0]["metadata"]["created_at"])

    def test_empty_content_writes_nothing(self, memory):
        memory.remember("", user_id="u1")
        assert all_rows(memory) == []


class TestRememberConversation:
    def test_stores_the_pair_as_one_row(self, memory):
        memory.remember_conversation("hi", "hello", user_id="u1")
        assert len(all_rows(memory)) == 1

    def test_uses_the_role_prefixed_format(self, memory):
        memory.remember_conversation("hi", "hello", user_id="u1")
        assert all_rows(memory)[0]["text"] == "user: hi\nassistant: hello"

    def test_embeds_the_combined_text(self, memory, fake_embedder):
        memory.remember_conversation("hi", "hello", user_id="u1")
        assert fake_embedder.document_calls[-1] == ["user: hi\nassistant: hello"]

    def test_tags_the_row_as_a_conversation(self, memory):
        memory.remember_conversation("hi", "hello", user_id="u1")
        assert all_rows(memory)[0]["metadata"]["type"] == "conversation"

    def test_scopes_the_row_to_the_user(self, memory):
        memory.remember_conversation("hi", "hello", user_id="u42")
        assert all_rows(memory)[0]["metadata"]["user_id"] == "u42"

    def test_does_not_invoke_the_splitter(self, conversation_store):
        """Conversation pairs are stored whole — splitting would break their coherence."""
        splitter = MagicMock()
        memory = LongTermMemory(conversation_store, text_splitter=splitter)
        memory.remember_conversation("x " * 2000, "y " * 2000, user_id="u1")
        splitter.split_text.assert_not_called()
        assert len(all_rows(memory)) == 1

    def test_multiline_content_round_trips(self, memory):
        memory.remember_conversation("line1\nline2", "a\nb", user_id="u1")
        assert all_rows(memory)[0]["text"] == "user: line1\nline2\nassistant: a\nb"


class TestRecall:
    def test_embeds_the_query(self, memory, fake_embedder):
        memory.recall("what do I like?", user_id="u1")
        assert fake_embedder.query_calls[-1] == "what do I like?"

    def test_filters_by_user_id(self, memory):
        memory.remember_conversation("mine", "yes", user_id="u1")
        memory.remember_conversation("theirs", "no", user_id="u2")
        texts = [r["text"] for r in memory.recall("anything", user_id="u1")]
        assert texts == ["user: mine\nassistant: yes"]

    def test_default_top_k_is_five(self, memory):
        for i in range(8):
            memory.remember_conversation(f"q{i}", f"a{i}", user_id="u1")
        assert len(memory.recall("q", user_id="u1")) == 5

    def test_custom_top_k(self, memory):
        for i in range(8):
            memory.remember_conversation(f"q{i}", f"a{i}", user_id="u1")
        assert len(memory.recall("q", user_id="u1", top_k=2)) == 2

    def test_returns_the_text_metadata_score_shape(self, memory):
        """The prompt builder and the agent tool read exactly these keys."""
        memory.remember_conversation("hi", "hello", user_id="u1")
        hit = memory.recall("hi", user_id="u1")[0]
        assert set(hit) == {"text", "metadata", "score"}
        assert isinstance(hit["score"], float)

    def test_no_memories_returns_empty_list(self, memory):
        assert memory.recall("anything", user_id="nobody") == []

    def test_recall_round_trips_a_remembered_conversation(self, memory):
        memory.remember_conversation("I am a pilot", "noted", user_id="u1")
        assert memory.recall("I am a pilot", user_id="u1")[0]["text"].startswith("user: I am a pilot")

    def test_closest_memory_comes_first(self, memory):
        memory.remember_conversation("unrelated topic entirely", "ok", user_id="u1")
        memory.remember_conversation("exact phrase", "ok", user_id="u1")
        # FakeEmbedder is deterministic, so the identical text is the nearest.
        top = memory.recall("user: exact phrase\nassistant: ok", user_id="u1")[0]
        assert "exact phrase" in top["text"]


class TestForgetUser:
    def test_removes_only_that_users_rows(self, memory):
        memory.remember_conversation("a", "b", user_id="u1")
        memory.remember_conversation("c", "d", user_id="u2")
        memory.forget_user("u1")
        assert [r["metadata"]["user_id"] for r in all_rows(memory)] == ["u2"]

    def test_recall_finds_nothing_after_forgetting(self, memory):
        memory.remember_conversation("a", "b", user_id="u1")
        memory.forget_user("u1")
        assert memory.recall("a", user_id="u1") == []

    def test_forgetting_an_unknown_user_is_a_no_op(self, memory):
        memory.remember_conversation("a", "b", user_id="u1")
        memory.forget_user("ghost")
        assert len(all_rows(memory)) == 1


class TestIdGeneration:
    def test_ids_are_prefixed_with_the_user(self, memory):
        memory.remember_conversation("a", "b", user_id="u1")
        assert all_rows(memory)[0]["id"].startswith("u1-")

    def test_ids_are_unique_across_calls(self, memory):
        for _ in range(5):
            memory.remember_conversation("same", "same", user_id="u1")
        ids = [r["id"] for r in all_rows(memory)]
        assert len(set(ids)) == 5

    def test_chunks_of_one_remember_call_get_distinct_ids(self, conversation_store):
        splitter = CharacterTextSplitter(separator=". ", chunk_size=20, chunk_overlap=0)
        memory = LongTermMemory(conversation_store, text_splitter=splitter)
        memory.remember("First fact here. Second fact here. Third fact here.", user_id="u1")
        ids = [r["id"] for r in all_rows(memory)]
        assert len(set(ids)) == 3
