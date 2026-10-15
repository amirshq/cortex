"""The application shell in src/api/main.py: /, /health, /metrics and CORS.

These endpoints sit outside the /api/v1 router, so no router test reaches
them. /metrics is what Prometheus scrapes; if it stops exposing a metric
family, the Grafana panels built on it go silently blank.
"""

from __future__ import annotations

import pytest

import src.api.router as router

pytestmark = pytest.mark.integration

CORTEX_METRIC_FAMILIES = (
    "http_requests_total",
    "http_request_duration_seconds",
    "http_requests_in_progress",
    "rate_limit_rejections_total",
    "chat_model_requests_total",
    "chat_tokens_total",
    "chat_cost_total",
    "embedding_requests_total",
    "embedding_cost_total",
    "rag_documents_indexed_total",
    "rag_chunks_indexed_total",
    "rag_queries_total",
    "rag_retrieval_top_score",
    "rag_retrieval_low_confidence_total",
)


class TestRootAndHealth:
    def test_root_returns_the_welcome_message(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the Cortex API!"}

    def test_health_reports_healthy(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_health_checks_are_never_rate_limited(self, client, fake_clock):
        """Orchestrators poll /health constantly; throttling it would get a
        healthy container killed."""
        capacity = router._rate_limiter.capacity

        statuses = {client.get("/health").status_code for _ in range(capacity * 2)}

        assert statuses == {200}
        assert router._rate_limiter.tokens == capacity


class TestMetricsEndpoint:
    def test_serves_the_prometheus_text_format(self, client):
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

    @pytest.mark.parametrize("family", CORTEX_METRIC_FAMILIES)
    def test_exposes_every_cortex_metric_family(self, client, family):
        assert f"# TYPE {family} " in client.get("/metrics").text

    def test_reflects_traffic_that_just_happened(self, client):
        client.get("/health")

        exposition = client.get("/metrics").text

        assert 'http_requests_total{method="GET",path="/health",status="200"}' in exposition


class TestCors:
    VITE_DEV_ORIGIN = "http://localhost:5173"

    def test_preflight_from_the_vite_dev_server_is_allowed(self, client):
        response = client.options("/api/v1/chat", headers={
            "Origin": self.VITE_DEV_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == self.VITE_DEV_ORIGIN
        assert "POST" in response.headers["access-control-allow-methods"]

    def test_cross_origin_responses_carry_an_allow_origin_header(self, client):
        response = client.get("/health", headers={"Origin": self.VITE_DEV_ORIGIN})

        assert response.headers["access-control-allow-origin"] in {"*", self.VITE_DEV_ORIGIN}
