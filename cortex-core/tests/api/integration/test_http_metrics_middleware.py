"""prometheus_middleware, observed through real requests.

The middleware wraps every request (except /metrics itself) and records:

    http_requests_total{method,path,status}        counter
    http_request_duration_seconds{method,path}     histogram
    http_requests_in_progress{method,path}         gauge

The registry is process-global and shared across the whole test run, so every
assertion compares values across a request — never an absolute value.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import src.api.router as router

pytestmark = pytest.mark.integration

COUNT = "http_requests_total"
DURATION_COUNT = "http_request_duration_seconds_count"
DURATION_SUM = "http_request_duration_seconds_sum"
IN_PROGRESS = "http_requests_in_progress"


class TestSuccessfulRequests:
    def test_counter_increments_once_with_method_path_and_status(self, client, metric):
        labels = {"method": "GET", "path": "/health", "status": "200"}
        before = metric(COUNT, labels)

        client.get("/health")

        assert metric(COUNT, labels) - before == 1

    def test_latency_is_observed_once_per_request(self, client, metric):
        labels = {"method": "GET", "path": "/health"}
        count_before, sum_before = metric(DURATION_COUNT, labels), metric(DURATION_SUM, labels)

        client.get("/health")

        assert metric(DURATION_COUNT, labels) - count_before == 1
        assert metric(DURATION_SUM, labels) > sum_before

    def test_in_progress_gauge_rises_during_the_request_and_recovers_after(self, client, metric, monkeypatch):
        labels = {"method": "POST", "path": "/api/v1/chat"}
        baseline = metric(IN_PROGRESS, labels)
        observed = {}

        async def business(request):
            observed["during"] = metric(IN_PROGRESS, labels)
            return {"reply": "ok", "model_used": "gpt-4o", "tokens_used": None}

        monkeypatch.setattr("src.api.controller.process_chat_message", business)

        assert client.post("/api/v1/chat", json={"message": "hi"}).status_code == 200
        assert observed["during"] == baseline + 1
        assert metric(IN_PROGRESS, labels) == baseline

    def test_the_response_passes_through_unmodified(self, client):
        response = client.get("/health")

        assert response.json() == {"status": "healthy"}
        assert response.headers["content-type"] == "application/json"


class TestErrorResponses:
    def test_a_handled_400_is_counted_with_status_400(self, client, metric):
        labels = {"method": "GET", "path": "/api/v1/sessions", "status": "400"}
        before = metric(COUNT, labels)

        assert client.get("/api/v1/sessions", params={"user_id": 0}).status_code == 400

        assert metric(COUNT, labels) - before == 1

    def test_a_controller_500_is_counted_with_status_500(self, client, metric, monkeypatch):
        monkeypatch.setattr("src.api.controller.query_rag",
                            AsyncMock(side_effect=RuntimeError("vector store unavailable")))
        labels = {"method": "POST", "path": "/api/v1/rag/query", "status": "500"}
        before = metric(COUNT, labels)

        assert client.post("/api/v1/rag/query", json={"question": "q?"}).status_code == 500

        assert metric(COUNT, labels) - before == 1

    def test_an_unhandled_exception_is_counted_as_500_and_the_gauge_recovers(self, client, metric, monkeypatch):
        """No controller catches this one: it propagates through call_next,
        so the middleware's finally block must still record it."""
        monkeypatch.setattr(router.chat_controller, "send_message",
                            AsyncMock(side_effect=RuntimeError("boom")))
        count_labels = {"method": "POST", "path": "/api/v1/chat", "status": "500"}
        gauge_labels = {"method": "POST", "path": "/api/v1/chat"}
        count_before, gauge_before = metric(COUNT, count_labels), metric(IN_PROGRESS, gauge_labels)

        assert client.post("/api/v1/chat", json={"message": "hi"}).status_code == 500

        assert metric(COUNT, count_labels) - count_before == 1
        assert metric(IN_PROGRESS, gauge_labels) == gauge_before


class TestScrapeEndpointIsExcluded:
    def test_requests_to_metrics_are_not_counted(self, client, metric):
        labels = {"method": "GET", "path": "/metrics", "status": "200"}
        before = metric(COUNT, labels)

        client.get("/metrics")
        client.get("/metrics")

        assert metric(COUNT, labels) - before == 0


# ---------------------------------------------------------------------------
# Known bug — asserts the CORRECT behaviour, so it is expected to fail today.
# ---------------------------------------------------------------------------
def _label_values_containing(fragment: str):
    from prometheus_client import REGISTRY

    return [sample for family in REGISTRY.collect() for sample in family.samples
            if fragment in sample.labels.get("path", "")]


class TestKnownIssuePathLabelCardinality:
    """KNOWN BUG. The middleware uses the raw request path as the `path`
    label. DELETE /sessions/{session_id} therefore creates a new time series
    per session id, and any unmatched URL (a scanner, a typo) creates one too
    — unbounded label cardinality, against the project's metric conventions.
    The fix is to label by route template (e.g. /api/v1/sessions/{session_id}).
    """

    @pytest.mark.xfail(strict=True, reason="KNOWN BUG: raw session ids become metric label values")
    def test_session_ids_do_not_become_label_values(self, client):
        client.delete("/api/v1/sessions/cardinality-probe-7f3a", params={"user_id": 0})

        assert _label_values_containing("cardinality-probe-7f3a") == []

    @pytest.mark.xfail(strict=True, reason="KNOWN BUG: unmatched paths become metric label values")
    def test_unmatched_paths_do_not_become_label_values(self, client):
        assert client.get("/api/v1/scanner-probe-91c2").status_code == 404

        assert _label_values_containing("scanner-probe-91c2") == []
