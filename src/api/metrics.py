"""Prometheus metrics for the personal chatbot API.

Exposes generic HTTP instrumentation (via `prometheus_middleware`) plus a few
business-level counters that controllers update directly. Scraped by Prometheus
at GET /metrics (see main.py) using the config in Docker/prometheus/.
"""

import time

from prometheus_client import Counter, Gauge, Histogram
from starlette.requests import Request

# --- HTTP-level metrics -----------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "path", "status"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently being processed",
    ["method", "path"],
)

# --- Business-level metrics --------------------------------------------------

RATE_LIMIT_REJECTIONS_TOTAL = Counter(
    "rate_limit_rejections_total",
    "Requests rejected by the token-bucket rate limiter",
)
CHAT_MODEL_REQUESTS_TOTAL = Counter(
    "chat_model_requests_total",
    "Successful chat requests handled, per LLM model",
    ["model"],
)
CHAT_TOKENS_TOTAL = Counter(
    "chat_tokens_total",
    "Tokens consumed by chat completions, per LLM model",
    ["model"],
)
RAG_DOCUMENTS_INDEXED_TOTAL = Counter(
    "rag_documents_indexed_total",
    "PDF documents indexed into the RAG vector store",
)
RAG_CHUNKS_INDEXED_TOTAL = Counter(
    "rag_chunks_indexed_total",
    "Text chunks indexed into the RAG vector store",
)
RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "RAG questions answered",
)
RAG_RETRIEVAL_TOP_SCORE = Histogram(
    "rag_retrieval_top_score",
    "Cross-encoder re-rank score of the top-ranked chunk per RAG query "
    "(higher = more confident match; the re-ranker's relevance gate is 0.15). "
    "A dropping median indicates retrieval quality is degrading.",
    buckets=(0, 0.15, 0.5, 1, 2, 3, 5, 8, 12),
)
RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL = Counter(
    "rag_retrieval_low_confidence_total",
    "RAG queries where the re-ranker found no chunk above the relevance gate "
    "and the pipeline fell back to raw vector-similarity order",
)


async def prometheus_middleware(request: Request, call_next):
    """ASGI middleware recording request count/latency/in-flight per (method, path).

    Paths in this API have no path parameters, so the raw request path is used
    directly as a label without a route-template lookup.
    """
    if request.url.path == "/metrics":
        return await call_next(request)

    method, path = request.method, request.url.path
    HTTP_REQUESTS_IN_PROGRESS.labels(method=method, path=path).inc()
    start = time.perf_counter()
    status_code = "500"
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        return response
    finally:
        duration = time.perf_counter() - start
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration)
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status_code).inc()
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, path=path).dec()
