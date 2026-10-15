# 📈 Prometheus + Grafana: system health, alerts, knowledge base

The monitoring stack already runs from `../../../Docker/` (Prometheus on 9090, Grafana on
3001, Alertmanager), and the app already exports metrics such as
`RAG_RETRIEVAL_LOW_CONFIDENCE_TOTAL`. That setup stays where it is, because Docker
provisions Grafana from `Docker/grafana/provisioning/`.

This folder holds the evaluation side of monitoring.

## What goes in this folder (planned)

| File | Purpose | Roadmap |
|---|---|---|
| `KNOWLEDGE_BASE.md` | Which knowledge-base metrics to export (document freshness from `indexed_at`, ingestion failures, index size) and how to read them | 5.6 |
| `check_metrics.py` | Scrapes `/metrics` from a running backend and checks the RAG metrics are present and sensible | 5.1 |

A new dashboard is designed and documented here, and its JSON is then copied to
`Docker/grafana/provisioning/dashboards/json/` so Grafana loads it.

## Status
◐ The stack runs; knowledge-base metrics and panels are to do.
