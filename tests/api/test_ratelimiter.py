"""Tests for the TokenBucket rate limiter."""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.api.ratelimiter import TokenBucket


class TestTokenBucketInitialization:
    """Test TokenBucket initialization and setup."""

    def test_initializes_with_full_capacity(self):
        """Bucket starts full."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        assert bucket.tokens == 10
        assert bucket.capacity == 10
        assert bucket.refill_rate == 2.0

    def test_initializes_with_zero_capacity(self):
        """Edge case: zero capacity."""
        bucket = TokenBucket(capacity=0, refill_rate=1.0)
        assert bucket.tokens == 0

    def test_initializes_with_fractional_refill_rate(self):
        """Refill rate can be fractional (e.g., 0.5 tokens/sec)."""
        bucket = TokenBucket(capacity=100, refill_rate=0.5)
        assert bucket.refill_rate == 0.5


class TestTokenBucketConsumption:
    """Test basic token consumption logic."""

    def test_consumes_single_token(self):
        """Can consume one token from full bucket."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        result = bucket.consume(1)
        assert result is True
        assert bucket.tokens == 9

    def test_consumes_multiple_tokens(self):
        """Can consume multiple tokens at once."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        result = bucket.consume(5)
        assert result is True
        assert bucket.tokens == 5

    def test_consumes_all_tokens(self):
        """Can consume exactly all tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        result = bucket.consume(10)
        assert result is True
        assert bucket.tokens == 0

    def test_rejects_when_insufficient_tokens(self):
        """Reject when requesting more tokens than available."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        bucket.consume(5)
        result = bucket.consume(6)
        assert result is False
        # Tokens may have been refilled slightly, so just check >= 5
        assert bucket.tokens >= 5

    def test_rejects_empty_bucket(self):
        """Reject request on empty bucket."""
        bucket = TokenBucket(capacity=1, refill_rate=0)
        bucket.consume(1)
        result = bucket.consume(1)
        assert result is False


class TestTokenBucketRefill:
    """Test token refill logic over time."""

    def test_refills_over_time(self):
        """Tokens refill as time passes."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        bucket.consume(5)  # Use 5 tokens
        assert bucket.tokens == 5

        time.sleep(0.1)  # 100ms pass
        bucket.consume(0)  # Trigger refill
        # Should have ~5 + (0.1 * 2.0) = 5.2 tokens
        assert bucket.tokens > 5

    def test_refill_respects_capacity(self):
        """Refill never exceeds capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)
        bucket.consume(5)
        time.sleep(1.0)
        bucket.consume(0)  # Trigger refill
        assert bucket.tokens == 10  # Capped at capacity

    def test_zero_refill_rate_stays_empty(self):
        """With refill_rate=0, tokens never increase."""
        bucket = TokenBucket(capacity=10, refill_rate=0)
        bucket.consume(5)
        time.sleep(0.2)
        bucket.consume(0)  # Trigger refill
        assert bucket.tokens == 5  # Still 5

    def test_fractional_token_accumulation(self):
        """Tokens accumulate as fractional values before consumption."""
        bucket = TokenBucket(capacity=100, refill_rate=0.5)
        bucket.consume(50)
        assert bucket.tokens == 50

        time.sleep(0.1)  # 100ms = 0.05 tokens at 0.5/sec
        bucket.consume(0)  # Trigger refill
        assert bucket.tokens > 50
        assert bucket.tokens < 51


class TestTokenBucketEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_consume_zero_tokens(self):
        """Can call consume(0) safely (doesn't change state)."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        result = bucket.consume(0)
        assert result is True  # Technically you always have 0+ tokens
        assert bucket.tokens == 10

    def test_negative_consumption_not_validated(self):
        """Negative tokens: the implementation doesn't validate input."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        bucket.consume(5)
        # Consume negative tokens would increase tokens (this is a quirk, not ideal)
        result = bucket.consume(-2)
        # With -2 tokens requested and 5 available, it's technically true
        # But the consume call subtracts -2, effectively adding 2
        assert result is True
        # Allow for slight refilling over time
        assert bucket.tokens >= 7  # At least 7 (5 - (-2) = 7)

    def test_large_capacity(self):
        """Works with large capacity values."""
        bucket = TokenBucket(capacity=1_000_000, refill_rate=1000.0)
        result = bucket.consume(500_000)
        assert result is True
        assert bucket.tokens == 500_000

    def test_very_small_refill_rate(self):
        """Works with very small refill rates."""
        bucket = TokenBucket(capacity=10, refill_rate=0.001)  # 1 token per 1000 seconds
        bucket.consume(5)
        time.sleep(0.1)
        bucket.consume(0)
        assert bucket.tokens > 5  # Minimal refill


class TestTokenBucketRealWorldScenarios:
    """Test realistic rate-limiting scenarios."""

    def test_steady_request_stream_under_limit(self):
        """Steady request stream stays under capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)  # 5 req/sec
        results = []
        for i in range(20):
            results.append(bucket.consume(1))
            time.sleep(0.05)  # 50ms between requests
        # Some requests might fail initially, but most should pass
        assert sum(results) > 10  # More than half should succeed

    def test_burst_then_wait_pattern(self):
        """Burst of requests, then wait, then burst again."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)  # 1 req/sec, capacity 5

        # First burst: consume all 5 tokens
        for _ in range(5):
            assert bucket.consume(1) is True

        # Now empty
        assert bucket.consume(1) is False

        # Wait for refill
        time.sleep(1.1)

        # Should have ~1 token refilled
        assert bucket.consume(1) is True
        assert bucket.consume(1) is False

    def test_request_at_exact_rate(self):
        """Requests arriving at exactly the refill rate succeed continuously."""
        bucket = TokenBucket(capacity=2, refill_rate=1.0)  # 1 token/sec
        bucket.consume(1)
        time.sleep(1.0)
        result = bucket.consume(1)
        assert result is True

    def test_api_rate_limit_scenario(self):
        """Typical API scenario: 60 req/min = 1 req/sec."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Immediate burst of 10 requests
        for _ in range(10):
            assert bucket.consume(1) is True

        # 11th request blocked
        assert bucket.consume(1) is False

        # After 1 second, 1 more request passes
        time.sleep(1.0)
        assert bucket.consume(1) is True


class TestTokenBucketConcurrency:
    """Test behavior under concurrent/rapid access (single-threaded simulation)."""

    def test_rapid_consume_calls_same_tick(self):
        """Multiple consume calls in same time instant."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        # All calls happen ~instantly, so refill time doesn't advance
        assert bucket.consume(3) is True
        assert bucket.consume(3) is True
        assert bucket.consume(3) is True
        assert bucket.consume(1) is True
        # Bucket should have 0 tokens left (10 - 3 - 3 - 3 - 1 = 0)
        assert bucket.tokens < 1
        assert bucket.consume(1) is False

    def test_refill_timestamp_updates_correctly(self):
        """Refill timestamp is updated after each refill."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        ts1 = bucket.last_refill_timestamp

        time.sleep(0.1)
        bucket.consume(0)  # Trigger refill
        ts2 = bucket.last_refill_timestamp

        assert ts2 > ts1

    def test_multiple_refills_accumulate(self):
        """Multiple time intervals correctly accumulate tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.consume(5)

        time.sleep(0.5)
        bucket.consume(0)
        tokens_at_half_sec = bucket.tokens  # ~5.5

        time.sleep(0.6)
        bucket.consume(0)
        tokens_at_one_sec = bucket.tokens  # Should be ~6.5 (or higher)

        assert tokens_at_one_sec > tokens_at_half_sec
