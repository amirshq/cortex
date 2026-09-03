# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## Project overview

Cortex is a learning-oriented FastAPI backend for a personal assistant chatbot,
paired with a small React (Vite) frontend. It follows a layered ("clean") architecture and
is intentionally heavily commented — the codebase doubles as a teaching resource for FastAPI
and LLM-app architecture patterns (see `src/Learn/`). Treat verbose docstrings/comments in
existing files as intentional; don't strip them out during unrelated edits.

Core capabilities:
- Chat endpoint backed by an LLM (OpenAI and/or Hugging Face models, configurable).
- Short-term conversation memory in Redis, longer-term/history storage in SQLite.
- RAG pipeline: PDF ingestion → chunking → Chroma vector store → retrieval → re-ranking.
- Token-bucket rate limiting on the API.
- **Agentic chatbot**: LLM with tool-calling loop for semantic memory recall and live web search.
- **Cost tracking**: Per-model LLM and embedding costs, visualized in Grafana dashboard.
- **Live data integration**: Web search and news retrieval via multiple providers (DuckDuckGo, NewsAPI).
- **Comprehensive test coverage**: Unit tests for all major components (rate limiter, LLM, embeddings, controllers, cost calculator, live data providers).

## Tech stack

- **Backend**: FastAPI 0.128, Uvicorn, SQLAlchemy 2.x, Pydantic DTOs.
- **LLM**: OpenAI API and/or Hugging Face Transformers (`LLM_PROVIDER` env var selects).
- **Memory**: Redis (short-term, TTL-based) + SQLite (`data/chatbot.db`).
- **Vector store**: Chroma (`chromadb`), embeddings via OpenAI or HF.
- **Frontend**: React 18 + Vite, in `src/ui/` (separate `package.json`).
- **Monitoring**: `prometheus-client` metrics exposed at `/metrics`, scraped by Prometheus,
  visualized in Grafana, alerted on via Alertmanager (all under `Docker/`).

## Renaming the Repository

This project is named **Cortex** (the application), but the GitHub repository is currently
named `personal-chatbot`. To rename the repository to `cortex`, follow these steps:

**On GitHub.com:**
1. Go to repository Settings → General
2. Scroll to "Repository name" and change it from `personal-chatbot` to `cortex`
3. Click "Rename"

**Locally (after renaming on GitHub):**
```bash
# Update your git remote URL
git remote set-url origin https://github.com/amirshq/cortex.git

# Optionally rename your local directory
mv ~/AI\ Projects/personal-chatbot ~/AI\ Projects/cortex
```

**Note:** The Docker container names and Prometheus job names will still use `personal-chatbot-*`
for consistency with monitoring configurations. Only the git repository name will change.

## Layout

```
src/
  api/            API layer: FastAPI app, router, controllers, rate limiter, metrics.
    main.py       App entrypoint, middleware, /, /health, /metrics.
    router.py     Route definitions -> controllers. Thin by design.
    controller.py Request/response handling, delegates to business layer.
    ratelimiter.py Token-bucket limiter (well-commented reference implementation).
    metrics.py    Prometheus metric definitions + HTTP instrumentation middleware.
  business/
    chatbot/      Chat orchestration (agentic_chatbot.py) with tool-calling loop.
    core/         model.py (LLM client), embedding.py, prompt_builder.py, cost.py, live_data.py.
    rag/          PDF ingestion, chunking, vector_store.py, retrieval.py, re_ranker/.
  database/
    dto.py        Pydantic request/response models (source of truth for API contracts).
    database.py   SQLAlchemy setup — currently EMPTY, not yet implemented.
  memory/
    redis_memory.py       Short-term memory (Redis lists, TTL).
    long_term_memory.py   Longer-term memory.
    chat_history_manager.py
    responsecache.py
  config/config.yml  Non-secret app config (model names, temperature, dirs, RAG params).
  utils/config.py    YAML loader for config.yml.
  ui/             React + Vite frontend (separate npm project).
  Learn/          Architecture/teaching docs — read before changing layer boundaries.
scripts/          CLI utilities (diagnose.py, index_cli.py, retrieval_cli.py).
tests/            pytest tests, currently only under tests/business/rag/.
Docker/           Dockerfile + docker-compose stack (app, redis, Prometheus, Alertmanager, Grafana, exporters).
data/             SQLite DB, Chroma persistence, RAG uploads — gitignored, do not commit.
```

## Running locally

```bash
# Backend (from repo root, with a venv active)
pip install -r requirements.txt
uvicorn src.api.main:app --reload

# Frontend
cd src/ui && npm install && npm run dev
```

Full stack (app + Redis + Prometheus + Alertmanager + Grafana) via Docker Compose:

```bash
cd Docker
docker compose up -d --build
```

- App: http://localhost:8000 (docs at `/docs`, metrics at `/metrics`)
- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093
- Grafana: http://localhost:3001 (mapped off 3000 to avoid clashing with the Vite dev
  server; default admin user `admin`, password from `GRAFANA_ADMIN_PASSWORD` in `.env`,
  defaults to `admin`)

The React UI (`src/ui/`) is not containerized — run it separately with `npm run dev`
(see below) regardless of whether the backend runs bare or via Docker Compose.

## Configuration

- Secrets and per-environment values live in `.env` (gitignored): `OPENAI_MODEL_NAME`,
  `OPENAI_TEMPERATURE`, `OPENAI_MAX_TOKENS`, `HF_TOKEN`, `LLM_PROVIDER`, plus `REDIS_URL`
  and `GRAFANA_ADMIN_PASSWORD` used by the Docker stack. Never commit real values — there
  is no `.env.example` yet; create one with placeholder values if you add new required
  variables.
- Non-secret tunables (model names, RAG `k`, history limits, agent config) live in
  `src/config/config.yml`, loaded via `src/utils/config.py`.

### Provider selection (on-prem vs. cloud)

Every swappable infra component is selected by its own env var, each defaulting to
today's on-prem/local behavior so existing deployments need zero config changes:

| Env var | Default (on-prem) | Alternative |
|---|---|---|
| `LLM_PROVIDER` | `openai` (cloud) | `huggingface` (fully local) · `azure_openai` |
| `EMBEDDING_PROVIDER` | `openai` | `azure_openai` |
| `VECTOR_STORE_PROVIDER` | `chroma` | `azure_search` — RAG chunks index |
| `CHAT_VECTOR_STORE_PROVIDER` | `chroma` | `azure_search` (not yet implemented) — conversation-memory index |
| `MEMORY_PROVIDER` | `redis` | `azure_redis` |
| `LIVE_DATA_PROVIDER` | `mock` (demo data) | `duckduckgo` (web search) · `newsapi` (news search) |

Each factory lives next to the classes it selects between — `create_llm()` in
`src/business/core/model.py`, `create_embedder()` in `src/business/core/embedding.py`,
`create_vector_store()` in `src/business/rag/vector_store.py`, `create_memory()` in
`src/memory/redis_memory.py`, `create_conversation_vector_store()` in
`src/memory/vectordb.py`. Requesting a not-yet-implemented provider raises a clear
`NotImplementedError` rather than silently falling back.

`azure_openai` (LLM + embeddings) is implemented — needs `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, and per-component deployment names
(`AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` / `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`) in
`.env`. `build_azure_openai_client()` in `model.py` is the shared client builder used by
both `create_llm()` and `AgenticChatbot`'s tool-calling loop (which needs a raw client,
not the `BaseLLM` wrapper, because it does OpenAI-style function calling). Note:
`AgenticChatbot` now raises `NotImplementedError` if `LLM_PROVIDER=huggingface` — it
always silently used OpenAI regardless of that setting before this was wired up, so this
is a deliberate small behavior change (fail loud instead of silently ignoring the
setting), not a regression.

`azure_redis` is implemented — needs `AZURE_REDIS_CONNECTION_STRING` in `.env`, a single
`rediss://:<access-key>@<name>.redis.cache.windows.net:6380/0` URL. Azure Cache for Redis
is Redis-protocol-compatible, so this reuses `RedisMemory` completely unchanged — the
factory just points it at a TLS URL instead of a plain `redis://` one and rejects
anything not using the `rediss://` scheme (Azure requires TLS).

`azure_search` for the RAG chunks index is implemented (`AzureSearchVectorStore` in
`vector_store.py`) — needs `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, and
`AZURE_SEARCH_INDEX_NAME`/`AZURE_SEARCH_EMBEDDING_DIM` (both have sensible defaults) in
`.env`. Needs Basic tier or above (Free tier has no vector search). The `azure-search-documents`
SDK is imported lazily inside the class (not at module level) so Chroma-only deployments
never need it installed, even though it's in `requirements.txt` unconditionally. The index
is created automatically on first use if missing; `reset()` drops and recreates it, matching
`ChromaVectorStore.reset()`'s semantics. Note: `query()` translates Azure Search's response
into the exact same nested-list dict shape Chroma's `.query()` returns (`{"ids": [[...]],
...}`) — `retrieval.py`'s `_retrieve()` depends on that specific structure regardless of
backend, so this translation layer is required, not incidental.

`CHAT_VECTOR_STORE_PROVIDER=azure_search` (conversation-memory index) is **not yet
implemented** — that's step 4b, a second index with a different schema
(`user_id`/`importance`/`created_at` instead of `source_id`/`chunk_start`/`section`).

RAG chunks and conversation memory deliberately use **two separate vector stores/indexes**
(`VECTOR_STORE_PROVIDER` vs. `CHAT_VECTOR_STORE_PROVIDER`) — they're different data with
different metadata shapes, not one index shared for two purposes.

### Live data integration

`LIVE_DATA_PROVIDER` selects how the chatbot fetches real-time information:

| Provider | Setup | Use case |
|---|---|---|
| `mock` (default) | None — returns synthetic data | Development/testing |
| `duckduckgo` | None — free API, no key needed | Web search, general queries |
| `newsapi` | Set `NEWS_API_KEY` env var (free at https://newsapi.org) | News-focused queries |

The `AgenticChatbot` class includes a `web_search` tool that the LLM can call when users
ask questions about current events, recent news, or real-time information. When the LLM detects
a web search is needed, it calls this tool and uses the results in its response.

Example `.env` setup:
```
LIVE_DATA_PROVIDER=newsapi
NEWS_API_KEY=your_free_newsapi_key_here
```

For on-prem deployments, `mock` is default (safe) and requires no external calls. To enable
live search, set `LIVE_DATA_PROVIDER=duckduckgo` (free, no key) or `newsapi` (with key).

### Cost tracking

LLM and embedding API costs are now tracked and visualized:

**Metrics** (`src/api/metrics.py`):
- `chat_cost_total{model}` — cumulative USD cost per LLM model (counter)
- `chat_model_requests_total{model}` — chat requests per model
- `embedding_cost_total` — cumulative USD cost of embeddings (counter)
- `embedding_requests_total` — embedding API calls

**Pricing** (`src/business/core/cost.py`):
- OpenAI and Azure OpenAI pricing tables (GPT-4o, GPT-3.5-turbo, etc.)
- Embedding pricing (text-embedding-3-small, text-embedding-3-large)
- Factory function `create_llm()` automatically calculates costs using token counts from LLM responses

**Grafana panels** (added to the "Cortex - Overview" dashboard):
- "Total LLM cost (USD)" — running sum
- "Total embedding cost (USD)" — running sum
- "Cost per request (avg)" — average cost per chat request
- "Embedding requests (total)" — cumulative embedding API calls
- "LLM cost over time by model" — trend line per model
- "Embedding cost over time" — trend line for all embeddings

Cost metrics are recorded only when token counts are available (most LLM APIs provide this).
Unknown models default to $0 cost to avoid breaking on new model names; log a warning in
production and add pricing as models are adopted.

## Testing

**Comprehensive test coverage** (run with `pytest` in venv):

- `tests/api/test_ratelimiter.py` — token-bucket rate limiter (initialization, consumption, refill, concurrency)
- `tests/api/test_controller.py` — HTTP controllers (send_message, get_history, query, upload)
- `tests/business/core/test_model.py` — LLM factory and implementations (OpenAI, Azure, HuggingFace)
- `tests/business/core/test_embedding.py` — embedding providers and factory
- `tests/business/core/test_cost.py` — cost calculation for LLM and embedding API calls
- `tests/business/core/test_live_data.py` — live data providers (mock, DuckDuckGo, NewsAPI)
- `tests/business/rag/` — retrieval, re-ranker, orchestrator (existing tests)

Run all tests: `python -m pytest tests/ -v`

Example queries to test live data integration:
- "What's in the news today?"
- "Tell me about recent AI developments"
- "Search for information about climate change"

The chatbot will automatically use the web_search tool when appropriate.

No lint/format tooling is configured (no ruff/black config found); match existing style by hand.

## Known gaps / things not to assume are wired up

- `src/database/database.py` is **empty** — SQLAlchemy engine/session setup is not yet
  implemented despite `dto.py` and `README.md` referencing it. Chat history persistence
  currently goes through `src/memory/` (Redis + long-term memory), not SQLAlchemy.
- `src/api/service.py` is empty — unused placeholder.
- `Docker/dockerfile` and `Docker/docker-compose.yml` were broken (invalid `COPY` syntax,
  empty compose file) before the monitoring stack work; they are now fixed as part of this
  change. If you see build failures referencing `COPY requirements.txt` with no destination,
  you're looking at stale docs/history, not the current file.
- Vector store artifacts under `src/business/rag/vectorstore/` and `data/` are build output,
  not source — don't hand-edit `.bin`/`.sqlite3` files.

## Conventions

- Keep the router thin: HTTP concerns in `router.py`/`controller.py`, business logic in
  `src/business/`. Follow the existing DTO pattern in `src/database/dto.py` for any new
  endpoint — don't accept raw `dict` request bodies.
- This repo favors explicit, heavily-commented code for pedagogical reasons. Match that
  tone in `src/` files that already have it; don't feel obligated to add the same density
  of comments to new infra/config files (Docker, Prometheus, Grafana) — keep those terse
  and operational.
- Prometheus metric names use the standard `_total` (counters) / `_seconds` (durations)
  suffixes; add labels sparingly (high-cardinality labels like raw user IDs or session IDs
  must never be used as metric labels).

## Monitoring & alerting

See `Docker/prometheus/alert_rules.yml` for current alert thresholds and
`Docker/grafana/provisioning/dashboards/json/api-overview.json` for the default dashboard (v2+).

**Metrics** exposed at `/metrics`:

**HTTP layer**:
- `http_requests_total{method,path,status}`, `http_request_duration_seconds{method,path}`,
  `http_requests_in_progress{method,path}` — generic HTTP instrumentation.

**Rate limiting**:
- `rate_limit_rejections_total` — requests rejected by token-bucket limiter.

**Chat/LLM**:
- `chat_model_requests_total{model}` — chat requests per LLM model.
- `chat_tokens_total{model}` — tokens consumed per model.
- `chat_cost_total{model}` — **NEW** — USD cost per model (from `src/business/core/cost.py`).

**Embeddings**:
- `embedding_requests_total` — **NEW** — embedding API calls.
- `embedding_cost_total` — **NEW** — USD cost of embeddings.

**RAG**:
- `rag_documents_indexed_total`, `rag_chunks_indexed_total` — ingestion volume.
- `rag_retrieval_top_score` (histogram) — re-ranker confidence (with 0.15 relevance gate).
- `rag_retrieval_low_confidence_total` — fallback to raw similarity (low confidence queries).
- `rag_queries_total` — queries answered.

Alertmanager's receiver is a placeholder (no Slack/email/PagerDuty wired up yet) — see the
comments in `Docker/alertmanager/alertmanager.yml` for how to add a real notification
channel before relying on this in anything beyond local dev.
