"""Tests for RedisMemory — short-term session memory — and its factory.

The Redis client is faked (an AsyncMock recording rpush/expire/lrange/
delete) so these run with no Redis server. What's under test is the
key scheme, the TTL refresh, the JSON round-trip, and the negative-index
window arithmetic in get_messages — all of which are easy to get subtly
wrong and invisible until a conversation loses its context.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.redis_memory import RedisMemory, ShortTermMemoryBase, create_memory


@pytest.fixture
def redis_client():
    client = MagicMock()
    client.rpush = AsyncMock()
    client.expire = AsyncMock()
    client.lrange = AsyncMock(return_value=[])
    client.delete = AsyncMock()
    return client


@pytest.fixture
def memory(redis_client) -> RedisMemory:
    with patch("src.memory.redis_memory.redis.from_url", return_value=redis_client):
        return RedisMemory(url="redis://localhost:6379/0")


class TestInterface:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            ShortTermMemoryBase()

    def test_subclass_must_implement_all_three_methods(self):
        class Incomplete(ShortTermMemoryBase):
            async def add_message(self, session_id, role, content): ...

        with pytest.raises(TypeError):
            Incomplete()

    def test_redis_memory_satisfies_the_interface(self, memory):
        assert isinstance(memory, ShortTermMemoryBase)


class TestInitialization:
    def test_decode_responses_is_enabled(self, redis_client):
        """Without it lrange returns bytes and json.loads gets bytes, not str."""
        with patch("src.memory.redis_memory.redis.from_url", return_value=redis_client) as from_url:
            RedisMemory(url="redis://x:6379/0")
        assert from_url.call_args.kwargs["decode_responses"] is True

    def test_default_ttl_is_one_hour(self, memory):
        assert memory.ttl == 3600

    def test_custom_ttl(self, redis_client):
        with patch("src.memory.redis_memory.redis.from_url", return_value=redis_client):
            assert RedisMemory(url="redis://x", ttl_seconds=60).ttl == 60


class TestAddMessage:
    @pytest.mark.asyncio
    async def test_uses_the_chat_prefixed_key(self, memory, redis_client):
        await memory.add_message("s1", "user", "hello")
        assert redis_client.rpush.call_args.args[0] == "chat:s1"

    @pytest.mark.asyncio
    async def test_stores_role_and_content_as_json(self, memory, redis_client):
        await memory.add_message("s1", "assistant", "hi there")
        payload = json.loads(redis_client.rpush.call_args.args[1])
        assert payload == {"role": "assistant", "content": "hi there"}

    @pytest.mark.asyncio
    async def test_appends_rather_than_replaces(self, memory, redis_client):
        """rpush, not set — conversation order depends on it."""
        await memory.add_message("s1", "user", "first")
        await memory.add_message("s1", "assistant", "second")
        assert redis_client.rpush.call_count == 2
        redis_client.set.assert_not_called() if hasattr(redis_client, "set") else None

    @pytest.mark.asyncio
    async def test_refreshes_the_ttl_on_every_write(self, memory, redis_client):
        """TTL must slide forward, otherwise an active conversation expires
        one hour after its FIRST message rather than its last."""
        await memory.add_message("s1", "user", "a")
        await memory.add_message("s1", "user", "b")
        assert redis_client.expire.call_count == 2
        assert redis_client.expire.call_args.args == ("chat:s1", 3600)

    @pytest.mark.asyncio
    async def test_unicode_survives_the_json_round_trip(self, memory, redis_client):
        await memory.add_message("s1", "user", "سلام 🎉")
        assert json.loads(redis_client.rpush.call_args.args[1])["content"] == "سلام 🎉"


class TestGetMessages:
    @pytest.mark.asyncio
    async def test_reads_the_last_n_with_negative_indices(self, memory, redis_client):
        """lrange(key, -limit, -1) is the sliding window. Off-by-one here
        silently drops the most recent turn from the model's context."""
        await memory.get_messages("s1", limit=10)
        assert redis_client.lrange.call_args.args == ("chat:s1", -10, -1)

    @pytest.mark.asyncio
    async def test_default_limit_is_ten(self, memory, redis_client):
        await memory.get_messages("s1")
        assert redis_client.lrange.call_args.args[1] == -10

    @pytest.mark.asyncio
    async def test_deserialises_into_dicts(self, memory, redis_client):
        redis_client.lrange.return_value = [
            json.dumps({"role": "user", "content": "q"}),
            json.dumps({"role": "assistant", "content": "a"}),
        ]
        assert await memory.get_messages("s1") == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

    @pytest.mark.asyncio
    async def test_empty_session_returns_empty_list(self, memory, redis_client):
        """The agent treats [] as "cold session" and hydrates from SQLite."""
        redis_client.lrange.return_value = []
        assert await memory.get_messages("missing") == []

    @pytest.mark.asyncio
    async def test_preserves_stored_order(self, memory, redis_client):
        redis_client.lrange.return_value = [
            json.dumps({"role": "user", "content": f"m{i}"}) for i in range(5)
        ]
        assert [m["content"] for m in await memory.get_messages("s1")] == \
            [f"m{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_round_trips_with_add_message(self, memory, redis_client):
        await memory.add_message("s1", "user", "hello")
        stored = redis_client.rpush.call_args.args[1]
        redis_client.lrange.return_value = [stored]

        assert await memory.get_messages("s1") == [{"role": "user", "content": "hello"}]


class TestClear:
    @pytest.mark.asyncio
    async def test_deletes_the_session_key(self, memory, redis_client):
        await memory.clear("s1")
        redis_client.delete.assert_called_once_with("chat:s1")


class TestCreateMemoryFactory:
    def test_default_provider_is_redis(self, monkeypatch):
        monkeypatch.delenv("MEMORY_PROVIDER", raising=False)
        with patch("src.memory.redis_memory.redis.from_url"):
            assert isinstance(create_memory(), RedisMemory)

    def test_default_url(self, monkeypatch):
        monkeypatch.delenv("MEMORY_PROVIDER", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        with patch("src.memory.redis_memory.redis.from_url") as from_url:
            create_memory()
        assert from_url.call_args.args[0] == "redis://localhost:6379/0"

    def test_reads_redis_url_from_env(self, monkeypatch):
        monkeypatch.delenv("MEMORY_PROVIDER", raising=False)
        monkeypatch.setenv("REDIS_URL", "redis://cache:6379/2")
        with patch("src.memory.redis_memory.redis.from_url") as from_url:
            create_memory()
        assert from_url.call_args.args[0] == "redis://cache:6379/2"

    def test_explicit_url_beats_env(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://env:6379/0")
        with patch("src.memory.redis_memory.redis.from_url") as from_url:
            create_memory(url="redis://explicit:6379/0")
        assert from_url.call_args.args[0] == "redis://explicit:6379/0"

    def test_provider_name_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MEMORY_PROVIDER", "  REDIS  ")
        with patch("src.memory.redis_memory.redis.from_url"):
            assert isinstance(create_memory(), RedisMemory)

    def test_ttl_is_passed_through(self, monkeypatch):
        monkeypatch.delenv("MEMORY_PROVIDER", raising=False)
        with patch("src.memory.redis_memory.redis.from_url"):
            assert create_memory(ttl_seconds=120).ttl == 120

    def test_azure_redis_reuses_redis_memory(self, monkeypatch):
        """Azure Cache for Redis is protocol-compatible — same class, TLS URL."""
        monkeypatch.setenv("MEMORY_PROVIDER", "azure_redis")
        monkeypatch.setenv("AZURE_REDIS_CONNECTION_STRING",
                           "rediss://:key@name.redis.cache.windows.net:6380/0")
        with patch("src.memory.redis_memory.redis.from_url") as from_url:
            memory = create_memory()

        assert isinstance(memory, RedisMemory)
        assert from_url.call_args.args[0].startswith("rediss://")

    def test_azure_redis_requires_a_connection_string(self, monkeypatch):
        monkeypatch.setenv("MEMORY_PROVIDER", "azure_redis")
        monkeypatch.delenv("AZURE_REDIS_CONNECTION_STRING", raising=False)
        with pytest.raises(RuntimeError, match="AZURE_REDIS_CONNECTION_STRING"):
            create_memory()

    def test_azure_redis_rejects_plaintext_scheme(self, monkeypatch):
        """Azure requires TLS on 6380. A redis:// URL would fail at connect
        time with an opaque timeout instead of a clear config error."""
        monkeypatch.setenv("MEMORY_PROVIDER", "azure_redis")
        monkeypatch.setenv("AZURE_REDIS_CONNECTION_STRING",
                           "redis://name.redis.cache.windows.net:6379/0")
        with pytest.raises(RuntimeError, match="rediss://"):
            create_memory()

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("MEMORY_PROVIDER", "memcached")
        with pytest.raises(ValueError, match="Unknown MEMORY_PROVIDER"):
            create_memory()
