"""Tests for ResponseCache — the Redis-backed LLM response cache.

The class is entirely a key-derivation problem, and both failure modes are
expensive:
  - keys too coarse  → one user is served another user's cached answer, or
                       a gpt-4o answer is returned for a gpt-3.5 request
  - keys too fine    → nothing ever hits, and the cache silently costs
                       money instead of saving it

So most of these assert on the exact key that gets built.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.responsecache import DEFAULT_TTL, ResponseCache


@pytest.fixture
def redis_client():
    client = MagicMock()
    client.get = AsyncMock(return_value=None)
    client.setex = AsyncMock()
    client.delete = AsyncMock()
    return client


@pytest.fixture
def cache(redis_client) -> ResponseCache:
    return ResponseCache(redis=redis_client)


class TestInitialization:
    def test_default_ttl_is_fifteen_minutes(self, cache):
        assert cache.ttl == DEFAULT_TTL == 900

    def test_custom_ttl(self, redis_client):
        assert ResponseCache(redis=redis_client, ttl=60).ttl == 60


class TestHashing:
    def test_same_payload_hashes_identically(self):
        payload = {"message": "hello"}
        assert ResponseCache._make_hash(payload) == ResponseCache._make_hash(payload)

    def test_different_payloads_hash_differently(self):
        assert ResponseCache._make_hash({"message": "hello"}) != \
               ResponseCache._make_hash({"message": "hi"})

    def test_key_order_does_not_affect_the_hash(self):
        """sort_keys=True — otherwise semantically identical requests miss."""
        assert ResponseCache._make_hash({"a": 1, "b": 2}) == \
               ResponseCache._make_hash({"b": 2, "a": 1})

    def test_hash_is_a_sha256_hex_digest(self):
        assert len(ResponseCache._make_hash({"m": "x"})) == 64

    def test_nested_payloads_are_hashable(self):
        assert ResponseCache._make_hash({"m": "x", "ctx": {"tz": "UTC", "k": [1, 2]}})

    def test_whitespace_differences_change_the_hash(self):
        assert ResponseCache._make_hash({"m": "hi"}) != ResponseCache._make_hash({"m": "hi "})


class TestKeyBuilding:
    def test_key_layout(self, cache):
        key = cache._build_key("u1", "gpt-4o", {"message": "hi"})
        parts = key.split(":")
        assert parts[0] == "response"
        assert parts[1] == "gpt-4o"
        assert parts[2] == "u1"
        assert len(parts[3]) == 64

    def test_different_users_get_different_keys(self, cache):
        """Cross-user cache bleed would leak private answers."""
        payload = {"message": "what is my address?"}
        assert cache._build_key("u1", "gpt-4o", payload) != \
               cache._build_key("u2", "gpt-4o", payload)

    def test_different_models_get_different_keys(self, cache):
        """A cheap model's answer must not be served as an expensive one's."""
        payload = {"message": "hi"}
        assert cache._build_key("u1", "gpt-4o", payload) != \
               cache._build_key("u1", "gpt-3.5-turbo", payload)

    def test_same_inputs_produce_a_stable_key(self, cache):
        a = cache._build_key("u1", "gpt-4o", {"message": "hi"})
        b = cache._build_key("u1", "gpt-4o", {"message": "hi"})
        assert a == b


class TestGet:
    @pytest.mark.asyncio
    async def test_miss_returns_none(self, cache, redis_client):
        redis_client.get.return_value = None
        assert await cache.get("u1", "gpt-4o", {"message": "hi"}) is None

    @pytest.mark.asyncio
    async def test_hit_returns_the_deserialised_response(self, cache, redis_client):
        redis_client.get.return_value = json.dumps({"reply": "cached answer"})
        assert await cache.get("u1", "gpt-4o", {"message": "hi"}) == {"reply": "cached answer"}

    @pytest.mark.asyncio
    async def test_reads_the_derived_key(self, cache, redis_client):
        await cache.get("u1", "gpt-4o", {"message": "hi"})
        assert redis_client.get.call_args.args[0] == \
            cache._build_key("u1", "gpt-4o", {"message": "hi"})

    @pytest.mark.asyncio
    async def test_empty_cached_string_is_treated_as_a_miss(self, cache, redis_client):
        redis_client.get.return_value = ""
        assert await cache.get("u1", "gpt-4o", {"message": "hi"}) is None


class TestSet:
    @pytest.mark.asyncio
    async def test_writes_with_the_ttl(self, cache, redis_client):
        await cache.set("u1", "gpt-4o", {"message": "hi"}, {"reply": "answer"})
        assert redis_client.setex.call_args.args[1] == 900

    @pytest.mark.asyncio
    async def test_stores_the_response_as_readable_json(self, cache, redis_client):
        """The answer is stored as JSON, not hashed — hashing it would make
        the cache write-only."""
        await cache.set("u1", "gpt-4o", {"message": "hi"}, {"reply": "answer"})
        assert json.loads(redis_client.setex.call_args.args[2]) == {"reply": "answer"}

    @pytest.mark.asyncio
    async def test_uses_the_same_key_as_get(self, cache, redis_client):
        """If set and get derived keys differently the hit rate would be 0."""
        payload = {"message": "hi"}
        await cache.set("u1", "gpt-4o", payload, {"reply": "a"})
        await cache.get("u1", "gpt-4o", payload)
        assert redis_client.setex.call_args.args[0] == redis_client.get.call_args.args[0]

    @pytest.mark.asyncio
    async def test_round_trip(self, cache, redis_client):
        payload = {"message": "hi"}
        await cache.set("u1", "gpt-4o", payload, {"reply": "stored"})
        redis_client.get.return_value = redis_client.setex.call_args.args[2]
        assert await cache.get("u1", "gpt-4o", payload) == {"reply": "stored"}

    @pytest.mark.asyncio
    async def test_custom_ttl_is_honoured(self, redis_client):
        cache = ResponseCache(redis=redis_client, ttl=42)
        await cache.set("u1", "m", {"q": "x"}, {"reply": "y"})
        assert redis_client.setex.call_args.args[1] == 42


class TestInvalidateUser:
    @pytest.mark.asyncio
    async def test_scans_with_a_user_scoped_pattern(self, cache, redis_client):
        async def scan_iter(match=None):
            for key in []:
                yield key

        redis_client.scan_iter = scan_iter
        await cache.invalidate_user("u1")

    @pytest.mark.asyncio
    async def test_deletes_every_matched_key(self, cache, redis_client):
        keys = ["response:gpt-4o:u1:aaa", "response:gpt-3.5:u1:bbb"]

        async def scan_iter(match=None):
            assert match == "response:*:u1:*"
            for key in keys:
                yield key

        redis_client.scan_iter = scan_iter
        await cache.invalidate_user("u1")

        assert [c.args[0] for c in redis_client.delete.call_args_list] == keys

    @pytest.mark.asyncio
    async def test_no_matches_deletes_nothing(self, cache, redis_client):
        async def scan_iter(match=None):
            for key in []:
                yield key

        redis_client.scan_iter = scan_iter
        await cache.invalidate_user("u1")
        redis_client.delete.assert_not_called()
