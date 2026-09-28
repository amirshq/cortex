# Cortex

**One personal-assistant chatbot, built twice — once by hand, once with an agent framework.**

Cortex is a chatbot with conversation memory, retrieval-augmented generation (RAG) over your
own PDFs, and an agentic tool-calling loop (semantic memory recall + live web search). This
repository holds two implementations of the *same* assistant, so the two approaches can be
compared side by side on identical features, tests, and evaluation data.

| Project | Approach | Status |
|---|---|---|
| [`cortex-core/`](cortex-core/) | Plain Python — agent loop, memory, and RAG pipeline written by hand, no agent framework | ✅ Working |
| [`cortex-langgraph/`](cortex-langgraph/) | The same assistant rebuilt with **LangChain + LangGraph** | 🚧 Just started |

## Why two implementations?

`cortex-core` shows *how* an LLM application works under the hood: every step of the
tool-calling loop, memory fan-out, and retrieval pipeline is explicit, heavily commented code.
`cortex-langgraph` rebuilds the same behavior on a framework, to learn what the framework
gives you, what it hides, and what it costs — in code size, latency, and control.

## Repository layout

```
cortex/
├── README.md                 ← you are here
├── cortex-core/              ← plain-Python implementation
│   ├── src/
│   │   ├── api/              FastAPI app, router, controllers, rate limiter, metrics
│   │   ├── business/
│   │   │   ├── chatbot/      Agentic chatbot with a hand-written tool-calling loop
│   │   │   ├── core/         LLM client, embeddings, prompt builder, cost, live data
│   │   │   └── rag/          PDF ingestion, chunking, vector store, retrieval, re-ranking
│   │   ├── memory/           Short-term (Redis) and long-term memory, chat history
│   │   ├── database/         Pydantic DTOs (API contracts)
│   │   ├── config/           Non-secret app config (config.yml)
│   │   ├── ui/               React + Vite frontend
│   │   └── Learn/            Architecture / teaching docs
│   ├── tests/                Unit, API integration, E2E tests, and real-model evals
│   ├── scripts/              CLI tools (indexing, retrieval, diagnostics)
│   ├── Docker/               App + Redis + Prometheus + Alertmanager + Grafana stack
│   └── README.md             Full documentation for cortex-core
└── cortex-langgraph/         ← LangChain + LangGraph implementation (in progress)
```

## Cortex-core — built with

| Area | Technology |
|---|---|
| Language | Python 3.11 |
| Web API | FastAPI 0.128, Uvicorn, Pydantic 2 (request/response DTOs) |
| LLM | OpenAI API · Azure OpenAI · Hugging Face Transformers (local, PyTorch) — selected by `LLM_PROVIDER` |
| Agent | Hand-written ReAct-style tool-calling loop using OpenAI function calling (no agent framework) |
| Embeddings | OpenAI / Azure OpenAI embeddings |
| Vector store | ChromaDB (local) · Azure AI Search |
| PDF parsing & OCR | Docling (layout, tables) with RapidOCR — runs fully locally |
| Re-ranking | Cross-encoder `BAAI/bge-reranker-base` (Hugging Face) |
| Memory | Redis (short-term, TTL) · Azure Cache for Redis · Chroma (semantic long-term memory) · SQLite |
| Live data | DuckDuckGo search · NewsAPI · mock provider |
| Rate limiting | Custom token-bucket limiter |
| Cost tracking | Per-model token pricing for LLM and embedding calls |
| Frontend | React 18, Vite 5, react-markdown |
| Monitoring | Prometheus client, Prometheus, Grafana, Alertmanager, cAdvisor, redis_exporter |
| Deployment | Docker, Docker Compose |
| Testing | pytest, pytest-asyncio — hermetic unit/integration/E2E tests plus opt-in real-model evals |
| CLI tools | Typer |

## Cortex-langgraph — built with

The plan below mirrors each part of `cortex-core` with its framework equivalent. It is a
starting point and will change as the project is built.

| Area | Technology (planned) |
|---|---|
| Language | Python 3.11 |
| Web API | FastAPI, Uvicorn, Pydantic 2 — same API contract as cortex-core |
| LLM | LangChain chat models (`langchain-openai`: `ChatOpenAI` / `AzureChatOpenAI`) |
| Agent | LangGraph `StateGraph` — nodes and edges replace the hand-written tool-calling loop |
| Tools | LangChain `@tool` definitions (memory recall, web search) |
| Embeddings | LangChain embeddings (`OpenAIEmbeddings` / `AzureOpenAIEmbeddings`) |
| Vector store | `langchain-chroma` · Azure AI Search (`langchain-community`) |
| PDF parsing & OCR | Docling via `langchain-docling` loader |
| Retrieval & re-ranking | LangChain retrievers + cross-encoder re-ranker |
| Memory | LangGraph checkpointer (short-term, per thread) + LangGraph store (long-term) |
| Frontend | Reuse the cortex-core React UI (same API contract) |
| Observability | LangSmith tracing, plus Prometheus metrics like cortex-core |
| Testing | pytest — shares cortex-core's evaluation data for a like-for-like comparison |

## Getting started

Each project is self-contained — `cd` into it and follow its own README.

```bash
cd cortex-core
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

See [`cortex-core/README.md`](cortex-core/README.md) for configuration, the Docker
monitoring stack, the frontend, and the test suite.
