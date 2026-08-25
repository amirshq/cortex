# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## Project overview

Personal Chatbot is a learning-oriented FastAPI backend for a personal assistant chatbot,
paired with a small React (Vite) frontend. It follows a layered ("clean") architecture and
is intentionally heavily commented — the codebase doubles as a teaching resource for FastAPI
and LLM-app architecture patterns (see `src/Learn/`). Treat verbose docstrings/comments in
existing files as intentional; don't strip them out during unrelated edits.

Core capabilities:
- Chat endpoint backed by an LLM (OpenAI and/or Hugging Face models, configurable).
- Short-term conversation memory in Redis, longer-term/history storage in SQLite.
- RAG pipeline: PDF ingestion → chunking → Chroma vector store → retrieval → re-ranking.
- Token-bucket rate limiting on the API.

## Tech stack

- **Backend**: FastAPI 0.128, Uvicorn, SQLAlchemy 2.x, Pydantic DTOs.
- **LLM**: OpenAI API and/or Hugging Face Transformers (`LLM_PROVIDER` env var selects).
- **Memory**: Redis (short-term, TTL-based) + SQLite (`data/chatbot.db`).
- **Vector store**: Chroma (`chromadb`), embeddings via OpenAI or HF.
- **Frontend**: React 18 + Vite, in `src/ui/` (separate `package.json`).
- **Monitoring**: `prometheus-client` metrics exposed at `/metrics`, scraped by Prometheus,
  visualized in Grafana, alerted on via Alertmanager (all under `Docker/`).

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
    chatbot/      Chat orchestration (agentic_chatbot.py).
    core/         model.py (LLM client), embedding.py, prompt_builder.py.
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

## Testing

- `pytest` — only `tests/business/rag/` is populated today (retrieval + re-ranker tests,
  some with `mocks.py`). `tests/Readme.md` is a learning note about test layout, not a
  description of full CI — don't assume coverage exists elsewhere.
- No lint/format tooling is configured (no ruff/black config found); match existing style
  by hand.

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
`Docker/grafana/provisioning/` for the default dashboard. The app exposes:

- `http_requests_total{method,path,status}`, `http_request_duration_seconds{method,path}`,
  `http_requests_in_progress{method,path}` — generic HTTP instrumentation (`src/api/metrics.py`).
- `rate_limit_rejections_total` — incremented when the token-bucket limiter rejects a request.
- `chat_model_requests_total{model}`, `chat_tokens_total{model}` — per-model chat usage.
- `rag_documents_indexed_total`, `rag_chunks_indexed_total` — RAG ingestion volume.

Alertmanager's receiver is a placeholder (no Slack/email/PagerDuty wired up yet) — see the
comments in `Docker/alertmanager/alertmanager.yml` for how to add a real notification
channel before relying on this in anything beyond local dev.
