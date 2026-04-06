# Comprehensive learning guide: building a Chatbot + RAG backend from scratch

This guide is for developers who want to **implement each piece in a sensible order** when starting a new project. It aligns with the layered layout used here: **API → business (chatbot, core, RAG) → memory → config/database DTOs**, and points to existing docs where they already exist.

**Already in this repo:**

- [FastAPI_Layers.md](FastAPI_Layers.md) — request flow through layers  
- [CONTROLLER_GUIDE.md](CONTROLLER_GUIDE.md) — controller responsibilities and patterns  

Use this file as a **curriculum**: what each module is for, what to learn first, and how pieces depend on each other.

---

## 1. Mental model: one request, many responsibilities

A single chat or RAG request typically does all of the following:

1. **Transport** — HTTP, auth, rate limits, CORS  
2. **Contract** — JSON shapes validated as Pydantic models (DTOs)  
3. **Orchestration** — controller turns HTTP concerns into calls to domain code  
4. **Domain** — chat logic, tool use, retrieval, prompting  
5. **Infrastructure** — LLM client, vector DB, Redis, file storage  

**Rule of thumb:** keep **HTTP** in `api/`, **rules and pipelines** in `business/`, **I/O adapters** in `memory/` (or a dedicated `infrastructure/` if you prefer that name). That separation makes tests and swaps (e.g. another vector store) much easier.

---

## 2. Recommended build order (greenfield)

Build **vertically thin slices** early (health + one chat endpoint), then deepen. Order:

| Phase | What to build | Why this order |
|-------|----------------|----------------|
| **A** | Config loading (`config.yml` + a small loader) | Everything else reads model names, paths, flags |
| **B** | DTOs (`database/dto.py` or equivalent) | Fixes the API contract before you wire logic |
| **C** | FastAPI app + router + one controller method | Proves end-to-end path; OpenAPI docs appear |
| **D** | Minimal LLM wrapper (`business/core/model.py`) | Single place for API keys, retries, streaming later |
| **E** | Prompt builder (`business/core/prompt_builder.py`) | Keeps prompts out of controllers and routers |
| **F** | Chatbot orchestration (`business/chatbot/`) | Uses D + E; still no RAG required |
| **G** | Embeddings (`business/core/embedding.py`) | Needed for both RAG and semantic memory |
| **H** | Vector store + ingest (`business/rag/`, chunking, index builder) | Offline/index path before query path |
| **I** | Retrieval + optional rerank (`retrieval.py`, `re_ranker/`) | “Retrieve → rerank → prompt → generate” |
| **J** | Memory layer (`memory/`) | Redis for short context; vector DB for long recall |
| **K** | Rate limiting, upload endpoints, CLI scripts | Hardening and ops |

You do **not** need RAG to ship a first chat; you **do** need config + DTOs + a thin API before anything else is testable.

---

## 3. Module-by-module: what it is and how to learn it

### 3.1 Config (`src/config/`, `src/utils/config.py`)

**Purpose:** One place for non-secret defaults (paths, collection names, model IDs) and environment-specific overrides.

**Learn:**

- YAML or `.env` for secrets; never commit API keys  
- Load once at startup; pass config into constructors (**dependency injection**) instead of importing globals everywhere  

**Practice:** Change only `config.yml` and confirm the app picks up a new `model_name` without code edits.

---

### 3.2 DTOs / validation (`src/database/dto.py`)

**Purpose:** Define the **HTTP contract**: request bodies, query params, and response shapes. Pydantic validates types and generates OpenAPI.

**Learn:**

- `BaseModel`, `Field`, optional vs required fields  
- Request DTOs for POST bodies; response models for stable JSON output  
- Keep “business-only” types internal; expose only what clients need  

**Practice:** Add a field to a response DTO and watch `/docs` update automatically.

**Related:** [FastAPI_Layers.md](FastAPI_Layers.md) (DTO layer).

---

### 3.3 API entry (`src/api/main.py`)

**Purpose:** Create the FastAPI `app`, attach CORS and middleware, mount routers, expose `/health`.

**Learn:**

- `FastAPI()` lifecycle vs per-request code  
- `include_router(..., prefix="/api/v1")` for versioning  

**Practice:** Add a trivial `GET /version` that returns a string from config.

---

### 3.4 Router (`src/api/router.py`)

**Purpose:** Map URLs and HTTP methods to **thin** handlers that delegate to controllers. Optionally attach **dependencies** (e.g. rate limiting).

**Learn:**

- `APIRouter`, `response_model=...`, `Depends()`  
- Keep routers free of business rules (“if message empty” belongs in controller or domain, not route decorators beyond validation)  

**Practice:** Add a new route that only calls one controller method and returns a DTO.

---

### 3.5 Controller (`src/api/controller.py`)

**Purpose:** **Translate** between HTTP and domain: check request-level rules, call business code, map exceptions to status codes, wrap results in response DTOs.

**Learn:**

- Thin controllers: no SQL/vector calls inlined if you can avoid it  
- `HTTPException` for 4xx/5xx; map domain errors deliberately  

**Related:** [CONTROLLER_GUIDE.md](CONTROLLER_GUIDE.md).

---

### 3.6 Rate limiting (`src/api/ratelimiter.py`)

**Purpose:** Protect the API from abuse; return **429** when exceeded.

**Learn:**

- Token bucket vs fixed window  
- Global vs per-user buckets (this project uses a shared bucket on routes; production often keys by IP or user id)  

**Practice:** Lower capacity in dev and trigger 429 from the client to see headers and body.

---

### 3.7 Business — core (`src/business/core/`)

**Typical pieces:**

| File | Role |
|------|------|
| `model.py` | LLM client wrapper (chat completions, temperature, system prompt injection) |
| `embedding.py` | Text → vectors for retrieval and memory |
| `prompt_builder.py` | Build system/user messages from templates + context |

**Learn:**

- **One** place to add logging, timeouts, and model name resolution  
- Prompts as functions or templates, not string concatenation scattered in five files  
- Embeddings: batch size, truncation, and dimension consistency with the vector index  

---

### 3.8 Business — chatbot (`src/business/chatbot/`)

**Purpose:** **Conversation orchestration**: history, tools (function calling), memory reads/writes, and calling the LLM.

**Learn:**

- **Dependency injection**: pass `LongTermMemory`, `RedisMemory`, etc., into the chatbot class — eases tests and swapping backends  
- **Tool definitions** (OpenAI-style `tools`) as data describing what the model may call  
- Where to persist a turn (after response) so the next turn can retrieve it  

**Pattern in this repo:** short-term Redis + long-term vector-backed memory + optional `ChatHistoryManager`.

---

### 3.9 Business — RAG (`src/business/rag/`)

RAG is a **pipeline**; learn it as sequential stages:

1. **Ingest** — PDF/text → clean chunks (`pdfingest/`, `chunk.py`)  
2. **Index** — embed chunks + store in vector DB (`index_builder.py`, `vector_store.py`)  
3. **Query** — embed question → retrieve top-k → optional **rerank** → build prompt → LLM (`retrieval.py`, `re_ranker/`)  

**Learn:**

- Chunk size, overlap, and metadata (page, source file) for citations  
- “Retrieve many, rerank few” for better precision  
- `PromptBuilder` to inject only the selected context (avoid stuffing the whole corpus)  

**Practice:** Run ingest on a tiny PDF, query with a phrase that appears only once, verify the answer cites the right chunk metadata.

---

### 3.10 Memory (`src/memory/`)

**Purpose:** Adapters for **conversation state** and **semantic recall** — not HTTP, not Pydantic DTOs.

**Learn:**

- **Short-term** (Redis): recent turns, TTL, session keying  
- **Long-term** (vector store): embed user/assistant messages or summaries; query by semantic similarity  
- Clear interfaces so `AgenticChatbot` does not import Chroma/SQL details directly  

---

### 3.11 Scripts (`scripts/`)

**Purpose:** CLIs for indexing, diagnostics, retrieval smoke tests — **repeatable** ops without hitting HTTP.

**Learn:**

- Same business functions as the API, different entrypoint  
- Good for CI: index fixture data, run a retrieval assertion  

---

### 3.12 Tests (`tests/`)

**Learn:**

- Unit-test **business** and **reranker** logic without FastAPI  
- Integration tests: TestClient against router with mocked LLM/vector store  
- Smoke tests for embedding + retrieval when keys and data are available  

---

## 4. Cross-cutting skills (do these in parallel)

- **Async vs sync:** FastAPI handlers are often `async`; some SDKs are sync — know when to use `asyncio.to_thread` or a thread pool.  
- **Observability:** structured logs with `request_id`, timing around LLM and retrieval.  
- **Security:** validate uploads, scan file types, cap payload size, never log secrets.  
- **Idempotency:** for “ingest this file twice,” decide dedupe strategy (hash + metadata).  

---

## 5. Minimal “first week” checklist

- [ ] Config loads; health endpoint works  
- [ ] DTOs defined; `/docs` shows schemas  
- [ ] POST chat returns a deterministic stub (no LLM) to prove wiring  
- [ ] Swap stub for real LLM via `business/core/model.py`  
- [ ] One RAG path: index 1 document, query, see grounded answer  
- [ ] Rate limit returns 429 under load test  

---

## 6. Where to look in this repository

| Concern | Primary location |
|--------|-------------------|
| HTTP surface | `src/api/main.py`, `router.py`, `controller.py` |
| API shapes | `src/database/dto.py` |
| LLM + prompts + embeddings | `src/business/core/` |
| Chat orchestration | `src/business/chatbot/agentic_chatbot.py` |
| RAG pipeline | `src/business/rag/retrieval.py` and subpackages |
| Memory | `src/memory/` |
| YAML config | `src/config/config.yml`, `src/utils/config.py` |

---

## 7. Suggested reading order

1. [FastAPI_Layers.md](FastAPI_Layers.md) — full stack mental model  
2. [CONTROLLER_GUIDE.md](CONTROLLER_GUIDE.md) — controller in detail  
3. Skim `src/api/router.py` and `src/database/dto.py` side by side  
4. Trace one chat request: `router` → `controller` → `business/chatbot`  
5. Trace one RAG query: `router` → `rag_controller` → `RAGPipeline` in `retrieval.py`  

This order mirrors how you would **build** the system from an empty repo: contract and API first, then intelligence, then retrieval and memory.
