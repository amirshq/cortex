# Cortex

> **Note:** This project is named "Cortex" (the application). The GitHub repository is currently named "personal-chatbot" and can be renamed following [this guide](CLAUDE.md#renaming-the-repository) if desired.

A personal assistant chatbot built with **FastAPI** (layered/clean architecture backend), an
**LLM** (OpenAI and/or Hugging Face, switchable), a **RAG pipeline** over your own PDFs, and a
**React (Vite) frontend**. It's intentionally heavily commented — the codebase doubles as a
teaching resource for FastAPI and LLM-app architecture patterns (see `src/Learn/`).

## 🎯 What it does

- **Chat** — conversational endpoint backed by an LLM, with short-term memory in Redis and
  longer-term history storage.
- **RAG over PDFs** — upload a PDF, it's parsed (text + tables) by
  [Docling](https://github.com/docling-project/docling) fully locally (no external API or
  API key), chunked, embedded, and stored in a Chroma vector store; ask questions and get
  answers grounded in retrieved, re-ranked chunks.
- **Rate limiting** — token-bucket limiter protecting the API.
- **Monitoring** — Prometheus metrics, Grafana dashboard, Alertmanager alert rules, all
  running via Docker Compose alongside the app.

## 🏗️ Architecture

Layered ("clean") architecture — see [FastAPI Layers Guide](src/Learn/FastAPI_Layers.md) for
the full walkthrough.

![ChatGPT-like Chat System with Memory](src/Images/ChatGPT-like%20Chat%20System%20with%20Memory.png)

```
src/
  api/            API layer: FastAPI app, router, controllers, rate limiter, metrics.
    main.py       App entrypoint, middleware, /, /health, /metrics.
    router.py     Route definitions -> controllers (thin by design).
    controller.py Request/response handling, delegates to business layer.
    ratelimiter.py Token-bucket limiter (well-commented reference implementation).
    metrics.py    Prometheus metric definitions + HTTP instrumentation middleware.
  business/
    chatbot/      Chat orchestration (agentic_chatbot.py).
    core/         model.py (LLM client), embedding.py, prompt_builder.py.
    rag/          PDF ingestion (Docling), chunking, vector_store.py, retrieval.py, re_ranker/.
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
  ui/             React + Vite frontend (separate npm project, not containerized).
  Learn/          Architecture/teaching docs — read before changing layer boundaries.
scripts/          CLI utilities (diagnose.py, index_cli.py, retrieval_cli.py).
tests/            pytest tests, currently only under tests/business/rag/.
Docker/           Dockerfile + docker-compose stack (app, redis, Prometheus, Alertmanager, Grafana, exporters).
data/             SQLite DB, Chroma persistence, RAG uploads — gitignored, not committed.
```

## 🚀 Getting started

### Prerequisites

- Python 3.11
- Node.js (for the frontend)
- Docker + Docker Compose (for the full stack, including monitoring)

### Backend

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

App: http://localhost:8000 — interactive docs at `/docs`, health at `/health`, metrics at `/metrics`.

### Frontend

```bash
cd src/ui
npm install
npm run dev
```

Vite prints the local URL it binds to (this project pins a fixed dev port; check the
terminal output for the exact address). It proxies `/api` requests to the backend on
port 8000, so run the backend first.

### Full stack via Docker Compose

```bash
cd Docker
docker compose up -d --build
```

This brings up the app, Redis, and the full monitoring stack:

| Service | URL |
|---|---|
| App | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Alertmanager | http://localhost:9093 |
| Grafana | http://localhost:3001 (user `admin`, password from `GRAFANA_ADMIN_PASSWORD` in `.env`, defaults to `admin`) |

The React UI is **not** containerized — run it separately with `npm run dev` regardless of
whether the backend runs bare or via Docker Compose.

## ⚙️ Configuration

- **Secrets/env** — `.env` (gitignored, not committed): `OPENAI_API_KEY`, `OPENAI_MODEL_NAME`,
  `OPENAI_TEMPERATURE`, `OPENAI_MAX_TOKENS`, `HF_TOKEN`, `LLM_PROVIDER`, plus `REDIS_URL` and
  `GRAFANA_ADMIN_PASSWORD` used by the Docker stack.
- **Non-secret tunables** — `src/config/config.yml` (model names, RAG `k`, history limits,
  agent config), loaded via `src/utils/config.py`.

## 📝 API endpoints

All under `/api/v1`:

| Method & path | What it does |
|---|---|
| `POST /chat` | Send a chat message |
| `GET /history` | Retrieve chat history |
| `POST /rag/upload` | Upload a PDF and (re-)build the RAG vector index |
| `POST /rag/query` | Ask a question answered from the indexed PDFs |

Plus `GET /`, `GET /health`, and `GET /metrics` at the root.

## 🧪 Testing

```bash
pytest
```

Only `tests/business/rag/` is populated today (retrieval + re-ranker tests). No lint/format
tooling is configured — match existing style by hand.

## 📊 Monitoring & alerting

Prometheus scrapes the app (`/metrics`), Redis (via `redis-exporter`), and container stats
(via `cadvisor`). Alert rules live in `Docker/prometheus/alert_rules.yml` (API down, high
error rate, high latency, rate-limit abuse, Redis down, container CPU/memory) and route
through Alertmanager (`Docker/alertmanager/alertmanager.yml` — the receiver ships as a
placeholder with no Slack/email/PagerDuty wired up; see the comments there to add one).
Grafana auto-provisions a dashboard (`Docker/grafana/provisioning/dashboards/json/api-overview.json`)
covering request rate/latency/errors, rate limiting, chat usage by model, RAG ingestion
volume, and container resources.

## 📚 Learning resources

This project includes learning materials in `src/Learn/`:

- **[FastAPI Layers Guide](src/Learn/FastAPI_Layers.md)** — architecture explanation
- **[Controller Guide](src/Learn/CONTROLLER_GUIDE.md)** — controller implementation walkthrough
- **[Comprehensive Learning Guide](src/Learn/COMPREHENSIVE_LEARNING_GUIDE.md)** — broader
  concepts guide (some PDF-parsing examples there still reference the `unstructured` library
  the RAG pipeline used previously; the pipeline itself now runs on Docling — see
  `src/business/rag/pdfingest/README.md`)

## 🛠️ Technology stack

- **Backend**: FastAPI, Uvicorn, Pydantic DTOs
- **LLM**: OpenAI API and/or Hugging Face Transformers (`LLM_PROVIDER` env var selects)
- **Memory**: Redis (short-term, TTL-based) + longer-term history storage
- **RAG**: Docling (PDF parsing) → chunking → Chroma (vector store) → retrieval → re-ranking
- **Frontend**: React 18 + Vite
- **Monitoring**: Prometheus, Grafana, Alertmanager (Docker Compose)

## 🔄 Known gaps

- `src/database/database.py` is **empty** — SQLAlchemy engine/session setup isn't
  implemented yet. Chat history persistence currently goes through `src/memory/`
  (Redis + long-term memory), not SQLAlchemy.
- `src/api/service.py` is empty — unused placeholder.
- No `.env.example` yet — create one with placeholder values if you add new required variables.

---

**Note**: Intended for learning and can be used in production with minor changes. See
`src/Learn/` for detailed guides on each component.
