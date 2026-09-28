"""The rate limit as a client experiences it: HTTP → dependency → bucket → 429.

tests/api/test_ratelimiter.py proves TokenBucket's arithmetic in isolation.
Nothing there proves the bucket is wired to a route, that an empty bucket
becomes a 429, or that rejections are counted. Those are the tests here.

Every test uses the fake clock, so refill is controlled exactly and nothing
sleeps. Capacity and refill rate are read from the production bucket, so the
tests follow router.py if the limits change.
"""

from __future__ import annotations

import pytest

import src.api.router as router

pytestmark = pytest.mark.integration

REJECTIONS = "rate_limit_rejections_total"


def post_chat(client, **body):
    return client.post("/api/v1/chat", json={"message": "hello", **body})


def exhaust_bucket(client):
    for _ in range(router._rate_limiter.capacity):
        assert post_chat(client).status_code == 200


class TestRateLimitedChat:
    def test_a_request_is_allowed_while_tokens_remain(self, client, chat_business, fake_clock):
        assert post_chat(client).status_code == 200
        chat_business.assert_awaited_once()

    def test_each_request_spends_exactly_one_token_via_the_dependency(self, client, chat_business, fake_clock):
        capacity = router._rate_limiter.capacity

        post_chat(client)
        post_chat(client)

        assert router._rate_limiter.tokens == capacity - 2

    def test_the_request_after_capacity_is_rejected_with_429(self, client, chat_business, fake_clock):
        exhaust_bucket(client)

        response = post_chat(client)

        assert response.status_code == 429
        assert response.json() == {"detail": "Rate limit exceeded. Please slow down."}

    def test_a_rejected_request_never_reaches_the_controller(self, client, chat_business, fake_clock):
        exhaust_bucket(client)
        calls_before = chat_business.await_count

        assert post_chat(client).status_code == 429
        assert chat_business.await_count == calls_before

    def test_each_rejection_increments_the_rejection_counter(self, client, chat_business, fake_clock, metric):
        exhaust_bucket(client)
        before = metric(REJECTIONS)

        post_chat(client)
        post_chat(client)

        assert metric(REJECTIONS) - before == 2

    def test_allowed_requests_leave_the_rejection_counter_alone(self, client, chat_business, fake_clock, metric):
        before = metric(REJECTIONS)

        exhaust_bucket(client)

        assert metric(REJECTIONS) - before == 0


class TestRefillOverHttp:
    def test_one_refill_interval_restores_exactly_one_request(self, client, chat_business, fake_clock):
        exhaust_bucket(client)
        assert post_chat(client).status_code == 429

        fake_clock.advance(1.0 / router._rate_limiter.refill_rate)

        assert post_chat(client).status_code == 200
        assert post_chat(client).status_code == 429

    def test_less_than_one_interval_is_not_enough(self, client, chat_business, fake_clock):
        exhaust_bucket(client)

        fake_clock.advance(0.5 / router._rate_limiter.refill_rate)

        assert post_chat(client).status_code == 429

    def test_a_long_idle_period_refills_only_up_to_capacity(self, client, chat_business, fake_clock):
        capacity = router._rate_limiter.capacity
        exhaust_bucket(client)

        fake_clock.advance(3600)
        statuses = [post_chat(client).status_code for _ in range(capacity + 3)]

        assert statuses.count(200) == capacity
        assert statuses[capacity:] == [429, 429, 429]


class TestRateLimitScopeAsImplemented:
    """Documents the limiter's CURRENT scope — see the final report.

    Only POST /chat carries the dependency, and the bucket is one module-level
    instance shared by every caller. Both are design choices worth revisiting
    (/rag/query also calls the LLM; one noisy client throttles everyone). If
    either changes, these tests should change with it.
    """

    def test_other_routes_keep_working_when_the_chat_bucket_is_empty(
            self, client, chat_business, history_business, session_store, rag_query_business, fake_clock):
        exhaust_bucket(client)
        assert post_chat(client).status_code == 429

        assert client.get("/api/v1/history", params={"user_id": 1}).status_code == 200
        assert client.get("/api/v1/sessions", params={"user_id": 1}).status_code == 200
        assert client.post("/api/v1/rag/query", json={"question": "q?"}).status_code == 200

    def test_unlimited_routes_spend_no_tokens(self, client, rag_query_business, fake_clock):
        capacity = router._rate_limiter.capacity

        for _ in range(capacity * 2):
            assert client.post("/api/v1/rag/query", json={"question": "q?"}).status_code == 200

        assert router._rate_limiter.tokens == capacity

    def test_the_bucket_is_shared_across_users(self, client, chat_business, fake_clock):
        for _ in range(router._rate_limiter.capacity):
            assert post_chat(client, user_id=1).status_code == 200

        assert post_chat(client, user_id=2).status_code == 429


class TestMalformedRequestsAndTheLimiter:
    """DESIGN NOTE — current behaviour, flagged in the report for a decision.

    FastAPI resolves the rate-limit dependency before validating the body, so
    a request rejected with 422 has already spent a token. A client sending
    malformed JSON can therefore drain the shared bucket, after which valid
    requests from everyone get 429.
    """

    def test_a_422_request_still_spends_a_token(self, client, chat_business, fake_clock):
        capacity = router._rate_limiter.capacity

        assert client.post("/api/v1/chat", json={}).status_code == 422

        assert router._rate_limiter.tokens == capacity - 1

    def test_malformed_requests_can_exhaust_the_bucket_for_valid_ones(self, client, chat_business, fake_clock):
        for _ in range(router._rate_limiter.capacity):
            assert client.post("/api/v1/chat", json={}).status_code == 422

        assert post_chat(client).status_code == 429
        chat_business.assert_not_called()
