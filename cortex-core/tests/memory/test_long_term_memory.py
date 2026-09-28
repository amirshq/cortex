"""Tests for LongTermMemory — the semantic-recall layer over the vector store.

LongTermMemory owns memory *semantics*, not storage: what gets chunked,
what metadata is attached, and how a turn is serialised before embedding.
Its collaborators are injected, so these tests use the in-memory fakes and
assert on the rows that would have been written.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.memory.long_term_memory import LongTermMemory
from tests.conftest import FakeConversationVectorStore, FakeEmbedder


class SplittingChunker:
    """Splits on '|' so chunking behaviour is visible in assertions."""
    def split(self, text): return [p for p in text.split("|") if p]


class IdentityChunker:
    def split(self, text): return [text]


@pytest.fixture
def store():
    return FakeConversationVectorStore()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def memory(store, embedder):
    return LongTermMemory(vectordb=store, embedder=embedder, chunker=IdentityChunker())


class TestRemember:
    def test_writes_one_row(self, memory, store):
        memory.remember("a fact", user_id="u1")
        assert len(store.rows) == 1
        assert store.rows[0]["text"] == "a fact"

    def test_chunks_before_storing(self, store, embedder):
        memory = LongTermMemory(vectordb=store, embedder=embedder, chunker=SplittingChunker())
        memory.remember("one|two|three", user_id="u1")
        assert [r["text"] for r in store.rows] == ["one", "two", "three"]

    def test_embeds_each_chunk(self, store, embedder):
        memory = LongTermMemory(vectordb=store, embedder=embedder, chunker=SplittingChunker())
        memory.remember("one|two", user_id="u1")
        assert embedder.embed_calls == ["one", "two"]

    def test_attaches_the_documented_metadata(self, memory, store):
        memory.remember("fact", user_id="u1", memory_type="preference", importance=5)
        meta = store.rows[0]["metadata"]
        assert meta["user_id"] == "u1"
        assert meta["type"] == "preference"
        assert meta["importance"] == 5
        assert meta["created_at"]

    def test_defaults_type_and_importance(self, memory, store):
        memory.remember("fact", user_id="u1")
        assert store.rows[0]["metadata"]["type"] == "knowledge"
        assert store.rows[0]["metadata"]["importance"] == 1

    def test_created_at_is_iso_parseable(self, memory, store):
        memory.remember("fact", user_id="u1")
        datetime.fromisoformat(store.rows[0]["metadata"]["created_at"])

    def test_empty_chunk_list_writes_nothing(self, store, embedder):
        class EmptyChunker:
            def split(self, text): return []

        LongTermMemory(store, embedder, EmptyChunker()).remember("x", user_id="u1")
        assert store.rows == []


class TestRememberConversation:
    """The hot path — called after every chat turn."""

    def test_stores_the_pair_as_one_row(self, memory, store):
        """Chunking is skipped deliberately: splitting a Q/A pair breaks the
        semantic coherence that makes recall useful."""
        memory.remember_conversation("what is X?", "X is Y", user_id="u1")
        assert len(store.rows) == 1

    def test_uses_the_role_prefixed_format(self, memory, store):
        memory.remember_conversation("Q", "A", user_id="u1")
        assert store.rows[0]["text"] == "user: Q\nassistant: A"

    def test_embeds_the_combined_text(self, memory, embedder):
        memory.remember_conversation("Q", "A", user_id="u1")
        assert embedder.embed_calls == ["user: Q\nassistant: A"]

    def test_tags_the_row_as_a_conversation(self, memory, store):
        """recall() and any future type filter depend on this tag."""
        memory.remember_conversation("Q", "A", user_id="u1")
        assert store.rows[0]["metadata"]["type"] == "conversation"

    def test_scopes_the_row_to_the_user(self, memory, store):
        memory.remember_conversation("Q", "A", user_id="u42")
        assert store.rows[0]["metadata"]["user_id"] == "u42"

    def test_does_not_invoke_the_chunker(self, store, embedder):
        class ExplodingChunker:
            def split(self, text): raise AssertionError("chunker must not run")

        LongTermMemory(store, embedder, ExplodingChunker()).remember_conversation(
            "Q", "A", user_id="u1")

    def test_multiline_content_round_trips(self, memory, store):
        memory.remember_conversation("line1\nline2", "resp", user_id="u1")
        assert "line1\nline2" in store.rows[0]["text"]


class TestRecall:
    def test_embeds_the_query(self, memory, embedder):
        memory.recall("what do I like?", user_id="u1")
        assert embedder.embed_calls == ["what do I like?"]

    def test_filters_by_user_id(self, memory, store):
        """Cross-user leakage here would put one user's private history in
        another user's system prompt."""
        memory.remember_conversation("mine", "yes", user_id="u1")
        memory.remember_conversation("theirs", "no", user_id="u2")

        results = memory.recall("anything", user_id="u1")
        assert len(results) == 1
        assert "mine" in results[0]["text"]

    def test_default_top_k_is_five(self, store, embedder):
        for i in range(10):
            store.rows.append({"id": str(i), "text": f"m{i}", "metadata": {"user_id": "u1"}})
        memory = LongTermMemory(store, embedder, IdentityChunker())
        assert len(memory.recall("q", user_id="u1")) == 5

    def test_custom_top_k(self, store, embedder):
        for i in range(10):
            store.rows.append({"id": str(i), "text": f"m{i}", "metadata": {"user_id": "u1"}})
        memory = LongTermMemory(store, embedder, IdentityChunker())
        assert len(memory.recall("q", user_id="u1", top_k=2)) == 2

    def test_returns_the_text_metadata_score_shape(self, memory, store):
        """build_agentic_system_prompt() reads r["text"] from each result."""
        memory.remember_conversation("Q", "A", user_id="u1")
        result = memory.recall("q", user_id="u1")[0]
        assert set(result) == {"text", "metadata", "score"}

    def test_no_memories_returns_empty_list(self, memory):
        assert memory.recall("q", user_id="nobody") == []

    def test_recall_round_trips_a_remembered_conversation(self, memory):
        memory.remember_conversation("I work as a data engineer", "Noted", user_id="u1")
        assert "data engineer" in memory.recall("job", user_id="u1")[0]["text"]


class TestForgetUser:
    def test_deletes_by_user_filter(self, memory, store):
        memory.forget_user("u1")
        assert store.deleted_filters == [{"user_id": "u1"}]

    def test_removes_only_that_users_rows(self, memory, store):
        memory.remember_conversation("mine", "a", user_id="u1")
        memory.remember_conversation("theirs", "b", user_id="u2")

        memory.forget_user("u1")
        assert [r["metadata"]["user_id"] for r in store.rows] == ["u2"]

    def test_recall_finds_nothing_after_forgetting(self, memory):
        memory.remember_conversation("Q", "A", user_id="u1")
        memory.forget_user("u1")
        assert memory.recall("q", user_id="u1") == []


class TestIdGeneration:
    def test_ids_are_prefixed_with_the_user(self, memory):
        assert memory._build_id("u1").startswith("u1-")

    def test_ids_are_unique_across_calls(self, memory, store):
        """Colliding ids would make each new turn overwrite the previous one."""
        for i in range(20):
            memory.remember_conversation(f"Q{i}", f"A{i}", user_id="u1")
        assert len({r["id"] for r in store.rows}) == 20
