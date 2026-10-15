# 🟠 Langfuse: traces, live cost and latency, scores, feedback

Langfuse is the dashboard for everything the LLM side of Cortex does. It's open source and
self-hosted with Docker, so questions, book chunks and answers stay on this machine.

## Run it

```bash
# from cortex-langgraph/, once: create the secrets (gitignored)
cp tests/tools/langfuse/env.example tests/tools/langfuse/.env   # then fill every value

docker compose -f tests/tools/langfuse/docker-compose.yml --env-file tests/tools/langfuse/.env up -d
.venv/bin/python tests/tools/langfuse/smoke_test.py             # checks it stores a trace
open http://localhost:3002                                      # log in with LANGFUSE_ADMIN_* from .env

docker compose -f tests/tools/langfuse/docker-compose.yml --env-file tests/tools/langfuse/.env down     # stop (data kept)
docker compose -f tests/tools/langfuse/docker-compose.yml --env-file tests/tools/langfuse/.env down -v  # stop and DELETE all traces
```

## What runs

| Container | Role | Host port |
|---|---|---|
| `langfuse-web` | UI + API (the SDK sends traces here) | **127.0.0.1:3002** |
| `langfuse-worker` | Processes queued traces into ClickHouse | none |
| `postgres` | Users, projects, keys, prompts, datasets | none |
| `clickhouse` | Traces, observations, scores (analytics) | none |
| `redis` | Ingestion queue and cache | none |
| `minio` | S3 storage for raw events and media | 127.0.0.1:9092 |

Changes from Langfuse's official compose file (see the header of `docker-compose.yml`): the
UI moved to 3002, the internal services publish no host ports (the official file would clash
with Prometheus on 9090 and the local Redis on 6379), there are no default passwords, Langfuse's
telemetry is off, and the org, project `cortex-langgraph`, API keys and admin login are created
on first start.

The first start takes a few minutes: about 2–3 GB of images, then database migrations.

## What goes in this folder

| File | Purpose | Roadmap | Status |
|---|---|---|---|
| `docker-compose.yml` | Self-hosted Langfuse | 5.2 | ✅ |
| `env.example` | The secrets `.env` needs | 5.2 | ✅ |
| `smoke_test.py` | auth check → send a trace → read it back | 5.2 | ✅ |
| `push_scores.py` | Attaches scores (our judge, RAGAS) to the traces of an eval run | 5.3 | ⏳ |
| `upload_dataset.py` | Uploads the golden set as a Langfuse dataset | 4.5 / 5.5 | ⏳ |
| `run_experiment.py` | Runs the golden set as a Langfuse dataset experiment | 4.6 (optional) | ⏳ |

**App hook (in `src/`, not here; next step):** a Langfuse `CallbackHandler` passed in the
`config` of the RAG and agent graphs. Off by default, switched on with `LANGFUSE_*` settings in
the app's `.env`, so the correctness tests stay offline.

## Things to know

- **Langfuse v4 is "events only".** The old `/api/public/traces` endpoints are gone; read
  traces through `api.observations.get_many(trace_id=...)` (see `smoke_test.py`).
- **`REDIS_SOCKET_TIMEOUT_MS: "0"`.** With the default 30 s, the worker logged about 1.5
  "Socket timeout" errors a second on idle queues. Turning it off stopped them, and ingestion
  is unaffected.
- **It restarts with Docker** (`restart: unless-stopped`) and uses about 2.7 GB of RAM. Stop it
  with `down` when not needed.

## Status
◐ Server running (Langfuse v4, 6 containers) and verified with `smoke_test.py`: auth, send,
stored in ClickHouse, read back, kept across a restart. Tracing hook in the app: next step.
