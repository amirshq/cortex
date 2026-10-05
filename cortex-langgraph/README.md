# Cortex — LangGraph edition

The same personal-assistant chatbot as [`cortex-core`](../cortex-core/), rebuilt on
**LangChain** components and **LangGraph** orchestration. Same API, same UI, same
features — so the two implementations can be compared on equal footing.

- **Chat** — an agent that recalls past conversations (semantic memory) and searches the
  web when it needs current information. Short-term memory in Redis, long-term memory in
  Chroma, full history in SQLite.
- **RAG over PDFs** — upload a PDF; it's parsed locally by Docling (text, tables, OCR),
  chunked, embedded and stored in a vector store. Questions are answered from retrieved,
  re-ranked chunks, with sources.
- **Rate limiting, cost tracking, monitoring** — token-bucket limiter, per-model cost
  metrics, Prometheus + Grafana + Alertmanager via Docker Compose.

## What LangChain / LangGraph replaced

| Module | cortex-core (hand-written) | cortex-langgraph |
|---|---|---|
| Chat model | `BaseLLM` + `OpenAIModel` / `LocalHFModel` | LangChain `ChatOpenAI` · `AzureChatOpenAI` · `ChatHuggingFace` |
| Embeddings | `Embedder` + `OpenAIEmbedder` with manual retries | `OpenAIEmbeddings` · `AzureOpenAIEmbeddings` (+ optional `CacheBackedEmbeddings`) |
| PDF loading | walking Docling's object model by hand | `langchain_docling.DoclingLoader` (Docling still does the parsing) |
| Chunking | fixed 800-char window | `RecursiveCharacterTextSplitter` (800 / 100, cuts on paragraph → sentence → word) |
| Vector stores | own `VectorStoreBase` + Chroma/Azure adapters + response-shape translation | `langchain_chroma.Chroma` · `langchain_azure_ai` `AzureSearch` |
| Re-ranker model | tokenizer + torch forward pass by hand | `sentence-transformers` `CrossEncoder` behind LangChain's `BaseCrossEncoder` |
| Prompts | f-strings in `PromptBuilder` | `ChatPromptTemplate` / `PromptTemplate` |
| Agent tools | JSON schemas + `if tool_name == ...` dispatch | `@tool` functions + LangGraph `ToolNode` |
| **Agent loop** | `while True:` loop | **LangGraph `StateGraph`**: `agent ⇄ tools` via `tools_condition` |
| **RAG pipeline** | three function calls + `select_context()` policy | **LangGraph `StateGraph`**: `retrieve → rerank → generate`, with `fallback` / `refuse` branches |

**Not replaced (and why):**

- **FastAPI layer, DTOs, rate limiter, metrics, React UI** — not LLM concerns. The API
  contract is identical, so the same UI works with both backends.
- **Relevance gate** (`rerank_documents`) — LangChain's `CrossEncoderReranker` discards
  scores and has no threshold, and the threshold is the RAG path's hallucination firewall.
- **Short-term memory stays plain Redis**, not a LangGraph checkpointer. LangGraph's Redis
  checkpointer needs Redis with the JSON + Search modules. The Redis in Docker Compose,
  Homebrew and Azure Cache for Redis (Basic/Standard) doesn't have them.
- **SQLite chat history** — it backs the session and history endpoints.
- **Live-data providers** — LangChain's DuckDuckGo wrapper lives in the retired
  `langchain-community` package.

### The two graphs

```
Agent (src/business/chatbot/agentic_chatbot.py)

  START → agent ──tools_condition──▶ tools (ToolNode) ─┐
            ▲                                          │
            └──────────────────────────────────────────┘
          agent ──(plain answer)──▶ END

RAG (src/business/rag/retrieval.py)

  START → retrieve → rerank ──(chunks passed the gate)──────────────▶ generate → END
                        ├──(none passed, hybrid / fail_open)──▶ fallback ─┘
                        └──(none passed, fail_closed)─────────▶ refuse → END
```

### Behaviour differences from cortex-core

These are deliberate, and each is covered by a test:

1. **The agent loop is bounded.** cortex-core's `while True:` had no cap. Here the graph
   allows 10 tool rounds per turn (`MAX_TOOL_ROUNDS`), then raises `GraphRecursionError`.
2. **Cost tracking works.** LangChain reports token usage on every response, so chat
   requests now return token counts and `chat_cost_total` is recorded. cortex-core always
   returned `tokens_used=None`, so its cost branch never ran.
3. **Long-term memory persists across restarts.** cortex-core's
   `chromadb.Client(Settings(persist_directory=...))` is an in-memory client in chromadb
   1.x, so it never wrote to disk.
4. **Tables stay where they are in the document.** Docling exports them inline as Markdown,
   instead of appending all tables after the body text.
5. **Chunks end on natural boundaries** (paragraph, then sentence, then word), not mid-word.
6. **Azure Search uses a different index** (`rag-chunks-langchain` by default). LangChain's
   `AzureSearch` schema is not compatible with cortex-core's `rag-chunks` index.

Kept identical on purpose: every prompt's wording, the re-ranker's raw-logit score scale
(so `min_score` gates the same chunks), `top_k` / `top_n` / chunk sizes, and the model defaults.

## Layout

```
src/
  api/            FastAPI app, router, controllers, rate limiter, metrics   (unchanged)
  business/
    chatbot/      agentic_chatbot.py (LangGraph agent), tools.py (@tool functions)
    core/         model.py (chat model factory), embedding.py, prompt_builder.py,
                  cost.py, live_data.py
    rag/          retrieval.py (LangGraph RAG graph), index_builder.py, vector_store.py,
                  pdfingest/ (DoclingLoader + text splitter), re_ranker/
  database/dto.py Pydantic request/response models (API contract)
  memory/         redis_memory.py, long_term_memory.py (LangChain VectorStore),
                  vectordb.py (Chroma factory), chat_history_manager.py (SQLite)
  config/         config.yml — every model, the RAG system prompt, the agent cap
  ui/             React + Vite frontend (separate npm project)
scripts/          index_cli.py, retrieval_cli.py, diagnose.py
tests/            hermetic unit / integration / E2E tests + opt-in real-model evals
Docker/           app + Redis + Prometheus + Alertmanager + Grafana
```

## Getting started

This project has its **own virtual environment**. LangChain needs a much newer `openai`
package than cortex-core pins, so the two can't share one environment.

```bash
cd cortex-langgraph
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn src.api.main:app --reload     # http://localhost:8000  (/docs, /health, /metrics)
```

Frontend (same UI as cortex-core):

```bash
cd src/ui && npm install && npm run dev
```

Full stack with monitoring: `cd Docker && docker compose up -d --build`. Its container names
and ports are the same as cortex-core's, so run one stack at a time.

## Configuration

Same `.env` variables and provider switches as cortex-core. The switches are
`LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `VECTOR_STORE_PROVIDER`, `CHAT_VECTOR_STORE_PROVIDER`,
`MEMORY_PROVIDER` and `LIVE_DATA_PROVIDER`.

**To change a model, edit `src/config/config.yml` and nothing else:**

```yaml
models:
  chat:      { name: gpt-4o }                                     # the agent
  rag:       { name: gpt-4o-mini, temperature: 0.7, max_tokens: 512 }
  embedding: { name: text-embedding-3-small }                     # changing it requires re-indexing
  reranker:  { name: BAAI/bge-reranker-base }
```

No model name appears anywhere in the code. On Azure the deployment names in `.env` are used
instead, and the Hugging Face provider reads `HF_MODEL_NAME`. The same file holds the RAG
system prompt (`prompts.rag_system_role`) and the agent's tool-round cap
(`agent.max_tool_rounds`). Duplicate keys are rejected when the app starts.

## API endpoints

All under `/api/v1`: `POST /chat`, `GET /history`, `GET /sessions`,
`DELETE /sessions/{session_id}`, `POST /rag/upload`, `POST /rag/query`. Plus `/`, `/health`
and `/metrics`.

## Testing

```bash
.venv/bin/python -m pytest              # 657 hermetic tests + 10 known-bug xfails, ~20s
.venv/bin/python -m pytest -m eval -s   # 35 real-model evals (needs OPENAI_API_KEY, costs cents)
```

The fakes implement the real LangChain interfaces (`BaseChatModel`, `Embeddings`), and
Chroma runs for real in a temporary directory. See [tests/TESTING.md](tests/TESTING.md).
