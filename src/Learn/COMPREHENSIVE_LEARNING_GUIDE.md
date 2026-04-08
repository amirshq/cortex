# Developer Learning Guide: Building a Production Chatbot with RAG

This guide walks you through every component of this project in the order you should build them. Each section explains the concept, the pattern used here, and a minimal working example to get you started.

---

## Table of Contents

0. [Quick reference — curriculum summary](#0-quick-reference--curriculum-summary)
1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Configuration & Environment Setup](#2-configuration--environment-setup)
3. [Data Transfer Objects (DTOs)](#3-data-transfer-objects-dtos)
4. [Database Layer — SQLite Chat History](#4-database-layer--sqlite-chat-history)
5. [Core LLM Wrapper](#5-core-llm-wrapper)
6. [Core Embedding Model](#6-core-embedding-model)
7. [Core Prompt Builder](#7-core-prompt-builder)
8. [Memory Layer — Vector Database (ChromaDB)](#8-memory-layer--vector-database-chromadb)
9. [Memory Layer — Short-Term Memory (Redis)](#9-memory-layer--short-term-memory-redis)
10. [Memory Layer — Long-Term Semantic Memory](#10-memory-layer--long-term-semantic-memory)
11. [Memory Layer — Response Cache](#11-memory-layer--response-cache)
12. [Business Layer — Agentic Chatbot](#12-business-layer--agentic-chatbot)
13. [RAG — PDF Ingestion](#13-rag--pdf-ingestion)
14. [RAG — Text Chunking](#14-rag--text-chunking)
15. [RAG — Vector Store](#15-rag--vector-store)
16. [RAG — Re-Ranking](#16-rag--re-ranking)
17. [RAG — Retrieval Pipeline](#17-rag--retrieval-pipeline)
18. [API Layer — FastAPI Application](#18-api-layer--fastapi-application)
19. [API Layer — Rate Limiter](#19-api-layer--rate-limiter)
20. [API Layer — Router & Endpoints](#20-api-layer--router--endpoints)
21. [API Layer — Controller](#21-api-layer--controller)
22. [Frontend (React UI)](#22-frontend-react-ui)
23. [Putting It All Together — The Build Order](#23-putting-it-all-together--the-build-order)

---

## 0. Quick reference — curriculum summary

This section condenses [SUMMARY_LEARNING_GUIDE.md](SUMMARY_LEARNING_GUIDE.md): **mental model**, **tables**, and **Purpose / Learn / Practice** notes per module. Use it as an at-a-glance index; the numbered sections below go deeper.

### Mental model: one request, many responsibilities

A single chat or RAG request typically involves:

1. **Transport** — HTTP, auth, rate limits, CORS  
2. **Contract** — JSON shapes validated as Pydantic models (DTOs)  
3. **Orchestration** — controller turns HTTP concerns into calls to domain code  
4. **Domain** — chat logic, tool use, retrieval, prompting  
5. **Infrastructure** — LLM client, vector DB, Redis, file storage  

**Rule of thumb:** keep **HTTP** in `api/`, **rules and pipelines** in `business/`, **I/O adapters** in `memory/` (or a dedicated `infrastructure/` package). That separation makes tests and backend swaps easier.

### Greenfield build order (phased)

Build **vertical slices** early (health + one chat endpoint), then deepen. This table is the **phase view**; [§23](#23-putting-it-all-together--the-build-order) lists a **file-by-file** order for this repo.

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

### Module purposes (one line each)

| Module | **Purpose** | Primary location |
|--------|-------------|------------------|
| Config | One place for **non-secret** defaults (paths, collection names, model IDs) and environment-specific overrides. | `src/config/`, `src/utils/config.py` |
| DTOs / validation | Define the **HTTP contract**: request bodies, query params, response shapes; Pydantic validates and drives OpenAPI. | `src/database/dto.py` |
| API entry | Create the FastAPI `app`, attach CORS/middleware, mount routers, expose `/health`. | `src/api/main.py` |
| Router | Map URLs and methods to **thin** handlers; delegate to controllers; optional `Depends()` (e.g. rate limit). | `src/api/router.py` |
| Controller | **Translate** HTTP ↔ domain: request rules, call business code, map exceptions to status codes, wrap response DTOs. | `src/api/controller.py` |
| Rate limiting | Protect the API from abuse; return **429** when exceeded. | `src/api/ratelimiter.py` |
| Business — core | LLM wrapper, embeddings, prompt construction — **one** place for model calls and prompt templates. | `src/business/core/` |
| Business — chatbot | **Conversation orchestration**: history, tools (function calling), memory reads/writes, LLM calls. | `src/business/chatbot/` |
| Business — RAG | Ingest → index → query pipeline (retrieve, optional rerank, grounded generation). | `src/business/rag/` |
| Memory | Adapters for **conversation state** and **semantic recall** — not HTTP, not public DTOs. | `src/memory/` |
| Scripts | CLIs for indexing, diagnostics, smoke tests — **repeatable** ops without HTTP. | `scripts/` |
| Tests | Unit business logic; integration via TestClient; smoke tests when keys/data exist. | `tests/` |

### Business — core files

| File | Role |
|------|------|
| `model.py` | LLM client wrapper (chat completions, temperature, system prompt injection) |
| `embedding.py` | Text → vectors for retrieval and memory |
| `prompt_builder.py` | Build system/user messages from templates + context |

**Learn (core):** one place for logging, timeouts, and model resolution; prompts as functions/templates; embedding batch size and dimension **consistent** with the vector index.

### RAG as stages

1. **Ingest** — PDF/text → clean chunks (`pdfingest/`, `chunk.py`)  
2. **Index** — embed chunks + store (`index_builder.py`, `vector_store.py`)  
3. **Query** — embed question → retrieve top-k → optional **rerank** → prompt → LLM (`retrieval.py`, `re_ranker/`)  

**Learn (RAG):** chunk size, overlap, metadata for citations; “retrieve many, rerank few”; inject only selected context in the prompt.

### Practice ideas (quick wins)

| Area | Practice |
|------|----------|
| Config | Change only `config.yml` and confirm a new `model_name` is picked up without code edits. |
| DTOs | Add a field to a response DTO and watch `/docs` update. |
| API entry | Add `GET /version` returning a string from config. |
| Router | Add a route that calls one controller method and returns a DTO. |
| Rate limit | Lower capacity in dev and trigger **429** from the client. |
| RAG | Ingest a tiny PDF; query with a unique phrase; verify metadata/citations. |

### Where to look in this repository

| Concern | Primary location |
|--------|-------------------|
| HTTP surface | `src/api/main.py`, `router.py`, `controller.py` |
| API shapes | `src/database/dto.py` |
| LLM + prompts + embeddings | `src/business/core/` |
| Chat orchestration | `src/business/chatbot/agentic_chatbot.py` |
| RAG pipeline | `src/business/rag/retrieval.py` and subpackages |
| Memory | `src/memory/` |
| YAML config | `src/config/config.yml`, `src/utils/config.py` |

### Cross-cutting skills (parallel with build)

- **Async vs sync:** FastAPI handlers are often `async`; some SDKs are sync — know `asyncio.to_thread` or thread pools when needed.  
- **Observability:** structured logs with `request_id`, timing around LLM and retrieval.  
- **Security:** validate uploads, cap payload size, never log secrets.  
- **Idempotency:** for “ingest this file twice,” define dedupe (hash + metadata).  

### Minimal first-week checklist

- [ ] Config loads; health endpoint works  
- [ ] DTOs defined; `/docs` shows schemas  
- [ ] POST chat returns a deterministic stub (no LLM) to prove wiring  
- [ ] Swap stub for real LLM via `business/core/model.py`  
- [ ] One RAG path: index one document, query, see grounded answer  
- [ ] Rate limit returns 429 under load test  

### Suggested reading order (docs + code)

1. [FastAPI_Layers.md](FastAPI_Layers.md) — full-stack mental model  
2. [CONTROLLER_GUIDE.md](CONTROLLER_GUIDE.md) — controller in detail  
3. Skim `src/api/router.py` and `src/database/dto.py` side by side  
4. Trace one chat: `router` → `controller` → `business/chatbot`  
5. Trace one RAG query: `router` → `rag_controller` → `RAGPipeline` in `retrieval.py`  

This mirrors **building** from an empty repo: contract and API first, then intelligence, then retrieval and memory.

---

## 1. Project Architecture Overview

### The Pattern: Clean N-Tier Architecture

This project follows a strict layered architecture. Each layer only talks to the layer directly below it. This keeps your code testable and replaceable.

```
┌─────────────────────────────────────┐
│           UI (React)                │  ← User interacts here
├─────────────────────────────────────┤
│        API Layer (FastAPI)          │  ← Receives HTTP requests
│    router.py → controller.py        │
├─────────────────────────────────────┤
│      Business Logic Layer           │  ← Core application logic
│   chatbot/ | rag/ | core/           │
├─────────────────────────────────────┤
│        Memory / Data Layer          │  ← All persistence
│  SQLite | ChromaDB | Redis          │
└─────────────────────────────────────┘
```

### Why this structure?

- **API layer** should never contain business logic. It only parses requests and formats responses.
- **Business layer** should never know about HTTP. It receives plain Python objects and returns plain Python objects.
- **Data layer** should never contain business logic. It only reads and writes data.

This separation means you can swap FastAPI for gRPC, or SQLite for PostgreSQL, without touching your business logic.

---

## 2. Configuration & Environment Setup

### Concepts: `.env` files and YAML config

There are two kinds of configuration:

- **Secrets** (API keys, tokens) → stored in `.env`, never committed to git
- **Application settings** (model names, directory paths, limits) → stored in `config.yml`, committed to git

### The `.env` file

```bash
# .env — secrets only, never commit this
OPENAI_API_KEY=sk-proj-...
LLM_PROVIDER=openai
UNSTRUCTURED_API_KEY=...
```

Load it with `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env into os.environ

api_key = os.getenv("OPENAI_API_KEY")
```

### The `config.yml` file

```yaml
# src/config/config.yml
llm_config:
  chat_model: "gpt-4o"
  temperature: 0.7

directories:
  db_path: "data/chatbot.db"
  vectordb_dir: "data/vectordb"

chat_history_config:
  max_history_pairs: 2
```

### The config loader (`src/utils/config.py`)

```python
import yaml
from pathlib import Path

def load_config(path: str = "src/config/config.yml") -> dict:
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)
```

### Key concept: Why separate secrets from settings?

- `.env` files contain credentials that must NOT be in version control.
- `config.yml` contains tuning parameters that developers need to see and discuss in PRs.
- This pattern is industry standard (12-Factor App methodology).

---

## 3. Data Transfer Objects (DTOs)

**File:** `src/database/dto.py`

### Concept: What is a DTO?

A DTO (Data Transfer Object) is a typed container for data moving between layers — from the HTTP request into your business logic, and back out to the HTTP response. Pydantic makes this automatic in Python.

### Why Pydantic?

Pydantic validates data at the boundary of your system (the API). If a client sends `{"message": 123}` when you expect a string, Pydantic raises an error before any business logic runs. This prevents invalid data from ever reaching your core.

### Minimal example

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ChatMessageRequest(BaseModel):
    message: str
    user_id: Optional[int] = None
    session_id: Optional[str] = None

class ChatMessageResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None
    timestamp: datetime
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
```

FastAPI automatically:
- Parses the incoming JSON into `ChatMessageRequest`
- Validates all field types
- Returns a 422 error with a clear message if validation fails
- Serializes `ChatMessageResponse` back to JSON

### Common field types to know

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class Example(BaseModel):
    required_string: str                          # must be present, must be a string
    optional_string: Optional[str] = None         # can be absent or null
    bounded_int: int = Field(default=10, ge=1, le=100)  # 1 ≤ value ≤ 100
    items: List[str] = []                         # list, defaults to empty
```

---

## 4. Database Layer — SQLite Chat History

**File:** `src/memory/chat_history_manager.py`

### Concept: Why SQLite?

SQLite is a file-based relational database — no server to run, no setup required. It's perfect for:
- Chat history (read/write patterns are simple)
- Single-user or low-concurrency applications
- Local development

For production at scale, swap to PostgreSQL with the same SQL schema.

### Schema design

```sql
CREATE TABLE sessions (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT 'New conversation',
    created_at TEXT NOT NULL
);

CREATE TABLE messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    role       TEXT NOT NULL,    -- 'user' or 'assistant'
    content    TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

### Key pattern: Auto-create schema on startup

```python
import sqlite3
from pathlib import Path

class ChatHistoryManager:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT 'New conversation',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()
```

### Reading and writing messages

```python
import uuid
from datetime import datetime, timezone

class ChatHistoryManager:
    # ... (init above)

    def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
                (session_id, user_id, "New conversation", now)
            )
        return session_id

    def add_message(self, session_id: str, user_id: str, role: str, content: str):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, user_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, role, content, now)
            )

    def get_messages(self, session_id: str, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id LIMIT ?",
                (session_id, limit)
            ).fetchall()
        return [dict(row) for row in rows]
```

### Key concepts to learn

- `CREATE TABLE IF NOT EXISTS` — idempotent schema creation
- Parameterized queries (`?` placeholders) — **always** use these, never string formatting (prevents SQL injection)
- `sqlite3.Row` — makes query results accessible as `row["column_name"]`
- UUIDs as primary keys — avoids collisions across distributed systems

---

## 5. Core LLM Wrapper

**File:** `src/business/core/model.py`

### Concept: Why wrap the LLM?

You wrap the OpenAI client (or any LLM) in your own class so that:
1. The rest of your code never imports `openai` directly
2. You can swap providers (OpenAI → Anthropic → local model) by changing one file
3. You control retry logic, error handling, and logging in one place

### Pattern: Abstract base class + concrete implementations

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, question: str, context: str = "") -> str:
        ...

class OpenAIModel(BaseLLM):
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, question: str, context: str = "") -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        else:
            messages.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
```

### The OpenAI Chat Completions API — what you need to know

```python
# The messages array is the entire conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},  # Instructions
    {"role": "user",   "content": "What is the capital of France?"},  # User turn
    {"role": "assistant", "content": "Paris."},                        # Previous AI turn
    {"role": "user",   "content": "And Germany?"},                     # Current question
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.7,     # 0.0 = deterministic, 1.0 = creative
    max_tokens=1000,     # max output tokens
)

reply = response.choices[0].message.content
tokens_used = response.usage.total_tokens
```
### Sample of LLM Respose
```python 

response = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1710000000,
    "model": "gpt-4.1-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "RAG stands for Retrieval-Augmented Generation. It combines retrieval with LLMs."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 45,
        "completion_tokens": 20,
        "total_tokens": 65
    }
}

```
### Tool calling (for agentic behavior)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_vector_db",
            "description": "Search the user's long-term memory for relevant past conversations",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # model decides when to call tools
)

# Check if the model wants to call a tool
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name      # "search_vector_db"
    arguments = json.loads(tool_call.function.arguments)  # {"query": "..."}
    # Execute the function, then call the API again with the result
```

---

## 6. Core Embedding Model

**File:** `src/business/core/embedding.py`

### Concept: What is an embedding?

An embedding converts text into a fixed-length vector of numbers (e.g., 1536 numbers for `text-embedding-3-small`). Similar texts produce similar vectors. This enables semantic search: you can find text that *means* the same thing, even if the exact words differ.

```
"What is the weather like?" → [0.02, -0.15, 0.87, ...]  (1536 numbers)
"How's the climate today?"  → [0.03, -0.14, 0.85, ...]  (very similar!)
"The cat sat on the mat"    → [0.55,  0.31, -0.22, ...]  (very different)
```

### Implementation

```python
from openai import OpenAI
from typing import Union

class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model_name = model_name

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
        """Embed many documents in batches (API has input limits)."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            # Results come back sorted by index
            embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(embeddings)
        return all_embeddings
```

### Key gotcha: Always batch embeddings

The API has input token limits. For large document sets, always split into batches of 50-100 texts. Never call the API once per document in a loop — you'll hit rate limits.

---

## 7. Core Prompt Builder

**File:** `src/business/core/prompt_builder.py`

### Concept: Why a dedicated prompt builder?

Prompts are code. They should be versioned, testable, and separated from business logic. A prompt builder assembles the `messages` array for the OpenAI API from structured inputs.

### Building a system prompt for RAG

```python
from datetime import datetime

class PromptBuilder:
    def build_rag_messages(self, question: str, context_chunks: list[str]) -> list[dict]:
        context = "\n\n---\n\n".join(context_chunks)
        system = (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context. If the context does not contain enough information "
            "to answer, say so clearly. Do not make up information.\n\n"
            f"Context:\n{context}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": question},
        ]

    def build_agentic_system_prompt(self, user_info: dict, context: str = "") -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        parts = [
            f"You are a personal AI assistant. Current time: {now}.",
            f"User: {user_info.get('name', 'Unknown')}",
        ]
        if context:
            parts.append(f"\nRelevant past context:\n{context}")
        return "\n".join(parts)
```

### Building conversation history for the chat API

The OpenAI API is stateless — you must send the entire conversation every time. This is your responsibility:

```python
def build_chat_messages(
    self,
    system_prompt: str,
    history: list[dict],   # [{"role": "user", "content": "..."}, ...]
    current_message: str,
    max_pairs: int = 5,
) -> list[dict]:
    # Trim history to avoid exceeding context window
    recent = history[-(max_pairs * 2):]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent)
    messages.append({"role": "user", "content": current_message})
    return messages
```

---

## 8. Memory Layer — Vector Database (ChromaDB)

**File:** `src/memory/vectordb.py`

### Concept: What is a vector database?

A vector database stores embeddings (vectors) alongside metadata and original text. Its key operation is **approximate nearest-neighbor (ANN) search**: given a query vector, find the N most similar stored vectors. This enables semantic search.

ChromaDB is a local, file-based vector database — no external service needed.

### Core operations

```python
import chromadb
from chromadb.config import Settings

class VectorDB:
    def __init__(self, persist_directory: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict = None,  # metadata filter
    ) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
        # Unpack ChromaDB's nested response format
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id":       results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def upsert(self, ids, embeddings, documents, metadatas):
        """Add or update — safe to call repeatedly."""
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
```

### ChromaDB gotchas

1. **IDs must be strings** — ChromaDB requires string IDs, not integers
2. **`upsert` vs `add`** — Use `upsert` when re-indexing documents to avoid duplicate errors
3. **`where` filter syntax** — Uses MongoDB-style operators: `{"user_id": {"$eq": "123"}}`
4. **Cosine vs L2 distance** — Cosine similarity (`hnsw:space: cosine`) is better for text embeddings

---

## 9. Memory Layer — Short-Term Memory (Redis)

**File:** `src/memory/redis_memory.py`

### Concept: Why Redis for short-term memory?

Redis is an in-memory key-value store. It's ideal for short-term conversation context because:
- Sub-millisecond reads/writes
- Automatic TTL (time-to-live) expiry — sessions auto-delete after 1 hour
- No schema needed

### Implementation

```python
import redis
import json
from datetime import timedelta

class RedisMemory:
    def __init__(self, host: str = "localhost", port: int = 6379, ttl_seconds: int = 3600):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    def add_message(self, session_id: str, role: str, content: str):
        key = self._key(session_id)
        message = json.dumps({"role": role, "content": content})
        self.client.rpush(key, message)           # append to list
        self.client.expire(key, self.ttl)          # reset TTL on every message

    def get_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        key = self._key(session_id)
        raw = self.client.lrange(key, -limit, -1)  # last N messages
        return [json.loads(m) for m in raw]

    def clear_session(self, session_id: str):
        self.client.delete(self._key(session_id))
```

### Key Redis commands to know

| Command | Purpose |
|---------|---------|
| `SET key value EX 3600` | Store a value with 1-hour expiry |
| `GET key` | Retrieve a value |
| `RPUSH key value` | Append to a list |
| `LRANGE key -20 -1` | Get last 20 items from a list |
| `EXPIRE key seconds` | Set/reset TTL on a key |
| `DEL key` | Delete a key |

---

## 10. Memory Layer — Long-Term Semantic Memory

**File:** `src/memory/long_term_memory.py`

### Concept: Semantic memory vs. keyword search

Long-term memory combines the embedding model (Chapter 6) and vector database (Chapter 8) to create **semantic search over past conversations**. You search by meaning, not exact words.

```
User asks: "What did we discuss about my project?"
→ Embed query
→ Search ChromaDB for similar past exchanges
→ Return relevant conversation snippets
→ Inject into system prompt for context
```

### Implementation

```python
import uuid
from datetime import datetime, timezone

class LongTermMemory:
    def __init__(self, vector_db: VectorDB, embedder: OpenAIEmbedder):
        self.db = vector_db
        self.embedder = embedder

    def remember(self, user_id: str, user_message: str, assistant_reply: str):
        """Store a conversation exchange."""
        text = f"User: {user_message}\nAssistant: {assistant_reply}"
        embedding = self.embedder.embed_query(text)
        self.db.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "type": "conversation",
            }],
        )

    def recall(self, user_id: str, query: str, n_results: int = 3) -> list[str]:
        """Retrieve past conversations relevant to the query."""
        query_embedding = self.embedder.embed_query(query)
        results = self.db.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where={"user_id": {"$eq": user_id}},  # only this user's memories
        )
        return [r["document"] for r in results]
```

### How this connects to the agentic chatbot

The `AgenticChatbot` exposes `recall()` as a **tool** the LLM can call:

```python
# The LLM decides when to call this tool based on the conversation
tools = [{
    "type": "function",
    "function": {
        "name": "search_vector_db",
        "description": "Search long-term memory for relevant past conversations",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
}]
```

---

## 11. Memory Layer — Response Cache

**File:** `src/memory/responsecache.py`

### Concept: Caching LLM responses

LLM API calls are expensive and slow. If users ask the same or very similar questions, you can return cached answers instantly. Use a Redis cache with TTL.

```python
import hashlib
import json
import redis

class ResponseCache:
    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 900):  # 15 min
        self.client = redis_client
        self.ttl = ttl_seconds

    def _cache_key(self, question: str) -> str:
        # Normalize the question and hash it
        normalized = question.strip().lower()
        return "response_cache:" + hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, question: str) -> str | None:
        key = self._cache_key(question)
        return self.client.get(key)

    def set(self, question: str, answer: str):
        key = self._cache_key(question)
        self.client.setex(key, self.ttl, answer)
```

### Usage pattern in the business layer

```python
cached = cache.get(question)
if cached:
    return cached

answer = llm.generate(question, context)
cache.set(question, answer)
return answer
```

### Key learning: Hash-based cache keys

Never use the raw question string as a Redis key — it could be very long and contain special characters. Hash it with SHA-256 to get a fixed-length, safe key.

---

## 12. Business Layer — Agentic Chatbot

**File:** `src/business/chatbot/agentic_chatbot.py`

### Concept: Agentic vs. simple chatbot

A simple chatbot sends the user's message to the LLM and returns the answer. An **agentic chatbot** gives the LLM tools and lets it decide whether to use them before answering. This enables behaviors like:
- "Let me search your memory for relevant past context..."
- "Let me look that up before answering..."

### The agentic loop

```python
class AgenticChatbot:
    def __init__(
        self,
        long_term_memory: LongTermMemory,
        redis_memory: RedisMemory,
        chat_history_manager: ChatHistoryManager,
        user_id: str,
        max_tool_calls: int = 3,
    ):
        self.long_term_memory = long_term_memory
        self.redis_memory = redis_memory
        self.history_manager = chat_history_manager
        self.user_id = user_id
        self.max_tool_calls = max_tool_calls
        self.client = OpenAI()

    def chat(self, user_message: str, session_id: str) -> str:
        # 1. Build context from memory
        recent_history = self.redis_memory.get_messages(session_id)
        semantic_context = self.long_term_memory.recall(self.user_id, user_message)

        # 2. Build messages array
        system_prompt = self.prompt_builder.build_agentic_system_prompt(
            user_info={"id": self.user_id},
            context="\n".join(semantic_context),
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_message})

        # 3. Agentic loop — let the LLM use tools
        tool_calls_made = 0
        while tool_calls_made < self.max_tool_calls:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self._get_tools(),
                tool_choice="auto",
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                # Execute the tool and feed result back
                messages.append(choice.message)  # add assistant's tool call to history
                for tool_call in choice.message.tool_calls:
                    result = self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                tool_calls_made += 1
            else:
                # Model gave a final answer
                break

        final_reply = response.choices[0].message.content

        # 4. Persist the exchange
        self.redis_memory.add_message(session_id, "user", user_message)
        self.redis_memory.add_message(session_id, "assistant", final_reply)
        self.long_term_memory.remember(self.user_id, user_message, final_reply)
        self.history_manager.add_message(session_id, self.user_id, "user", user_message)
        self.history_manager.add_message(session_id, self.user_id, "assistant", final_reply)

        return final_reply

    def _execute_tool(self, tool_call) -> str:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        if name == "search_vector_db":
            results = self.long_term_memory.recall(self.user_id, args["query"])
            return "\n".join(results) if results else "No relevant memories found."
        return "Unknown tool."
```

### Key learning: Dependency Injection

`AgenticChatbot` never creates its own dependencies — they are passed in. This is **dependency injection**. Benefits:
- In tests, pass mock objects instead of real databases
- Swap `RedisMemory` for `InMemoryMemory` without changing the chatbot code
- Explicit dependencies — reading the constructor tells you everything the class needs

### The factory function pattern

```python
# src/business/chatbot/__init__.py
def _make_chatbot(user_id: str) -> AgenticChatbot:
    config = load_config()
    embedder = OpenAIEmbedder(config["vectordb_config"]["embedding_model"])
    vector_db = VectorDB(config["directories"]["vectordb_dir"], "chat_history")
    long_term_memory = LongTermMemory(vector_db, embedder)
    redis_memory = RedisMemory()
    history_manager = ChatHistoryManager(config["directories"]["db_path"])
    return AgenticChatbot(long_term_memory, redis_memory, history_manager, user_id)

async def process_chat_message(request: ChatMessageRequest) -> ChatMessageResponse:
    chatbot = _make_chatbot(str(request.user_id))
    reply = chatbot.chat(request.message, request.session_id)
    return ChatMessageResponse(reply=reply, session_id=request.session_id, ...)
```

---

## 13. RAG — PDF Ingestion

**File:** `src/business/rag/pdfingest/unstructure_pdf_digest.py`

### Concept: What is document ingestion?

Before you can search a PDF, you must:
1. **Parse** it — extract text, tables, and images from the PDF format
2. **Structure** the content — identify which parts are headings, body text, tables, etc.
3. **Normalize** it — clean up formatting artifacts, whitespace, etc.

The **Unstructured** library handles all of this automatically.

### Using Unstructured to parse PDFs

```python
from unstructured.partition.pdf import partition_pdf
from dataclasses import dataclass

@dataclass
class IngestedDocument:
    source_id: str        # filename
    text: str             # full extracted text
    metadata: dict        # page numbers, section types, etc.

def ingest_pdf(file_path: str) -> IngestedDocument:
    # Unstructured extracts and classifies all elements
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",           # higher accuracy, slower
        infer_table_structure=True,  # extract tables as structured data
        include_page_breaks=True,
    )
    
    # Elements can be: Title, NarrativeText, Table, Image, ListItem, etc.
    text_parts = []
    for element in elements:
        element_type = type(element).__name__
        if element_type in ("NarrativeText", "Title", "ListItem"):
            text_parts.append(str(element))
        elif element_type == "Table":
            text_parts.append(f"[TABLE]\n{element.metadata.text_as_html}\n[/TABLE]")
    
    return IngestedDocument(
        source_id=Path(file_path).name,
        text="\n\n".join(text_parts),
        metadata={"file": file_path, "pages": len(set(e.metadata.page_number for e in elements))},
    )
```

### Processing a directory of PDFs

```python
from pathlib import Path

def ingest_directory(directory: str) -> list[IngestedDocument]:
    docs = []
    for pdf_path in Path(directory).glob("*.pdf"):
        doc = ingest_pdf(str(pdf_path))
        docs.append(doc)
    return docs
```

---

## 14. RAG — Text Chunking

**File:** `src/business/rag/pdfingest/chunk.py`

### Concept: Why chunk documents?

LLMs have context window limits. A 200-page PDF won't fit in one prompt. More importantly, retrieving the *entire* PDF for every query is wasteful — you want the 2-3 most relevant paragraphs.

The solution is to split documents into overlapping chunks before indexing.

```
[... paragraph 1 ...][... paragraph 2 ...][... paragraph 3 ...]
         ↓ chunk with overlap
Chunk 1: [... paragraph 1 ......... paragraph 2 ...]
Chunk 2: [............. paragraph 2 ......... paragraph 3 ...]
                ↑ overlap ensures boundary context isn't lost
```

### Implementation

```python
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    source_id: str
    chunk_index: int
    start_char: int
    end_char: int

class Chunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: IngestedDocument) -> list[Chunk]:
        text = document.text
        chunks = []
        start = 0
        idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at a sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind(". ")
                if last_period > self.chunk_size // 2:
                    end = start + last_period + 1
                    chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    source_id=document.source_id,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                ))
                idx += 1
            
            start = end - self.overlap  # move back by overlap amount
        
        return chunks
```

### Key decisions when chunking

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| `chunk_size` | 500–1500 chars | Larger = more context per chunk, fewer results; Smaller = more precise retrieval |
| `overlap` | 50–200 chars | Larger = less risk of cutting context at boundaries |

---

## 15. RAG — Vector Store

**File:** `src/business/rag/vector_store.py`

### Concept

This is the RAG-specific vector store (separate from the chat memory vector store). It wraps the low-level `VectorDB` class with RAG-specific operations: indexing chunks and querying by similarity.

```python
from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    text: str
    source_id: str
    chunk_index: int
    score: float         # similarity score (0–1, higher is more similar)
    metadata: dict

class VectorStore:
    def __init__(self, persist_directory: str):
        self.db = VectorDB(persist_directory, collection_name="pdf_chunks")
        self.embedder = OpenAIEmbedder()

    def upsert(self, chunks: list[Chunk]):
        """Embed and store chunks."""
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        self.db.upsert(
            ids=[f"{c.source_id}_{c.chunk_index}" for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "source_id": c.source_id,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
            } for c in chunks],
        )

    def query(self, question: str, top_k: int = 30) -> list[RetrievedChunk]:
        """Retrieve top-k chunks most similar to the question."""
        query_embedding = self.embedder.embed_query(question)
        results = self.db.search(query_embedding, n_results=top_k)
        return [
            RetrievedChunk(
                text=r["document"],
                source_id=r["metadata"]["source_id"],
                chunk_index=r["metadata"]["chunk_index"],
                score=1 - r["distance"],  # convert cosine distance to similarity
                metadata=r["metadata"],
            )
            for r in results
        ]
```

---

## 16. RAG — Re-Ranking

**Files:** `src/business/rag/re_ranker/`

### Concept: Why re-rank?

Vector similarity search is fast but imprecise. It finds chunks that are topically related, but not necessarily the ones that directly answer the question.

**Re-ranking** uses a more powerful model (a cross-encoder) to re-score the top-K retrieved chunks against the query. The cross-encoder reads both the query and the chunk together, producing a much more accurate relevance score.

```
Vector search:  Retrieve top 30 chunks (fast, approximate)
                           ↓
Cross-encoder:  Re-score all 30 pairs (query, chunk) with a neural model
                           ↓
Re-ranker:      Sort by new scores, apply threshold filter, return top 5
```

### Cross-encoder vs. bi-encoder (embeddings)

| | Bi-encoder (embeddings) | Cross-encoder |
|---|---|---|
| **How** | Encode query and doc separately | Encode query + doc together |
| **Speed** | Fast (pre-compute doc vectors) | Slow (must run model per pair) |
| **Accuracy** | Lower (no cross-attention) | Higher (full attention over both) |
| **Use case** | First-stage retrieval (top 30) | Re-ranking (top 5 from 30) |

### Cross-encoder implementation

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        """Score each (query, chunk) pair."""
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)  # returns numpy array
        return scores.tolist()
```

### Re-ranker with gating (hallucination control)

Gating filters out chunks below a confidence threshold. This prevents the LLM from using weakly relevant context and hallucinating connections.

```python
class ReRanker:
    def __init__(
        self,
        cross_encoder: CrossEncoderReRanker,
        confidence_threshold: float = 0.5,
        top_k_input: int = 30,
        top_n_output: int = 5,
    ):
        self.cross_encoder = cross_encoder
        self.threshold = confidence_threshold
        self.top_k_input = top_k_input
        self.top_n_output = top_n_output

    def re_rank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        # 1. Limit input
        candidates = chunks[:self.top_k_input]

        # 2. Score
        scores = self.cross_encoder.score(query, candidates)

        # 3. Attach scores and apply gating
        scored = []
        for chunk, score in zip(candidates, scores):
            if score >= self.threshold:
                chunk.score = score
                scored.append(chunk)

        # 4. Sort by score and truncate
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:self.top_n_output]
```

### Context selection policy (fail-open vs fail-closed)

```python
def select_context(
    chunks: list[RetrievedChunk],
    fallback_chunks: list[RetrievedChunk],
    policy: str = "hybrid",
) -> tuple[list[RetrievedChunk], bool]:
    """
    policy options:
      "strict"  → return [] if no chunks pass gating (fail-closed, safest)
      "open"    → always return top-k vector results (fail-open, most helpful)
      "hybrid"  → use strict, but fall back to vector results if empty
    """
    if policy == "strict":
        return chunks, len(chunks) > 0
    elif policy == "open":
        return fallback_chunks, True
    else:  # hybrid
        if chunks:
            return chunks, True
        return fallback_chunks, False  # False = low confidence
```

---

## 17. RAG — Retrieval Pipeline

**File:** `src/business/rag/retrieval.py`

### Concept: Orchestrating the full RAG pipeline

The retrieval pipeline combines all RAG components into a single `answer(question)` method.

```python
class RAGPipeline:
    def __init__(self):
        config = load_config()
        self.vector_store = VectorStore(config["rag_vectorstore_dir"])
        cross_encoder = CrossEncoderReRanker()
        self.re_ranker = ReRanker(cross_encoder)
        self.llm = OpenAIModel(
            model_name=config["llm_config"]["rag_model"],
            temperature=0.0,  # deterministic for factual RAG
            max_tokens=1500,
        )
        self.prompt_builder = PromptBuilder()

    def answer(self, question: str) -> dict:
        # Step 1: Retrieve
        retrieved = self.vector_store.query(question, top_k=30)
        if not retrieved:
            return {"answer": "No relevant documents found.", "sources": []}

        # Step 2: Re-rank
        reranked = self.re_ranker.re_rank(question, retrieved)

        # Step 3: Select context (with fallback policy)
        final_chunks, high_confidence = select_context(reranked, retrieved[:5], policy="hybrid")

        # Step 4: Generate
        context_texts = [c.text for c in final_chunks]
        messages = self.prompt_builder.build_rag_messages(question, context_texts)
        
        response = self.llm.client.chat.completions.create(
            model=self.llm.model_name,
            messages=messages,
        )
        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": [{"text": c.text, "source": c.source_id, "score": c.score} for c in final_chunks],
            "high_confidence": high_confidence,
        }
```

---

## 18. API Layer — FastAPI Application

**File:** `src/api/main.py`

### Concept: Why FastAPI?

FastAPI is the de facto standard for Python APIs in 2024-2025. Key features:
- **Async-first** — handles many concurrent requests without blocking
- **Auto-generated docs** — OpenAPI docs at `/docs` for free
- **Pydantic integration** — request/response validation is automatic
- **Type hints = documentation** — function signatures are self-documenting

### Minimal FastAPI app

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Personal Chatbot API", version="1.0.0")

# CORS — required for browser-based frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # in production, specify exact origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check — always implement this
@app.get("/health")
async def health():
    return {"status": "ok"}

# Include routers — register sub-modules
from src.api.router import router
app.include_router(router, prefix="/api/v1")
```

### Running the app

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Async vs sync endpoints

```python
# Sync — blocks the event loop during execution
@app.post("/sync")
def sync_endpoint(request: ChatMessageRequest):
    result = some_slow_operation()
    return result

# Async — allows other requests while waiting for I/O
@app.post("/async")
async def async_endpoint(request: ChatMessageRequest):
    result = await some_async_operation()
    return result
```

**Rule of thumb:** Use `async def` for endpoints that do I/O (database queries, HTTP calls, file reads). Use `def` for CPU-bound work.

---

## 19. API Layer — Rate Limiter

**File:** `src/api/ratelimiter.py`

### Concept: Token Bucket Algorithm

A token bucket is a classic rate-limiting algorithm. The bucket holds tokens; each request consumes one. Tokens refill at a fixed rate. This allows short bursts while enforcing a steady-state rate.

```
Bucket capacity: 10 tokens
Refill rate:      2 tokens/second
1 request costs:  1 token

Scenario:
- Bucket starts full (10 tokens)
- 10 rapid requests: allowed (empties bucket)
- 11th immediate request: rejected (0 tokens)
- Wait 0.5 sec: 1 token added → 1 more request allowed
- Wait 5 sec: 10 tokens added (capped at 10) → burst allowed again
```

### Implementation

```python
import time
import threading

class TokenBucket:
    def __init__(self, capacity: int = 10, refill_rate: float = 2.0):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
```

### Using it as FastAPI middleware

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Create one bucket (shared across all requests)
_bucket = TokenBucket(capacity=10, refill_rate=2.0)

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _bucket.consume():
            raise HTTPException(status_code=429, detail="Too many requests")
        return await call_next(request)

app.add_middleware(RateLimitMiddleware)
```

---

## 20. API Layer — Router & Endpoints

**File:** `src/api/router.py`

### Concept: Why separate routes from the main app?

The `APIRouter` lets you define routes in a separate file, then register them all at once in `main.py`. For large APIs, you'd have one router per feature (chat, rag, auth, etc.).

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from src.database.dto import (
    ChatMessageRequest, ChatMessageResponse,
    ChatHistoryRequest, RAGQueryRequest, RAGQueryResponse,
)
from src.business.chatbot import process_chat_message, get_chat_history
from src.business.rag import query_rag, ingest_pdfs

router = APIRouter()

@router.post("/chat", response_model=ChatMessageResponse)
async def chat(request: ChatMessageRequest):
    return await process_chat_message(request)

@router.get("/history")
async def history(
    user_id: int = Query(...),
    session_id: str = Query(None),
    limit: int = Query(default=50, ge=1, le=100),
):
    return await get_chat_history(ChatHistoryRequest(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
    ))

@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    return await query_rag(request.question)

@router.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    # Save file, then index
    upload_dir = "data/rag_uploads"
    file_path = f"{upload_dir}/{file.filename}"
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    result = await ingest_pdfs(upload_dir)
    return result
```

### Common HTTP patterns

```python
# Path parameters
@router.get("/session/{session_id}")
async def get_session(session_id: str):
    ...

# Query parameters
@router.get("/messages")
async def get_messages(limit: int = Query(default=20), offset: int = Query(default=0)):
    ...

# Request body
@router.post("/chat")
async def chat(request: ChatMessageRequest):  # Pydantic model = body
    ...

# File upload
@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    ...

# Return specific HTTP status codes
from fastapi import status
@router.post("/create", status_code=status.HTTP_201_CREATED)
async def create(...):
    ...
```

---

## 21. API Layer — Controller

**File:** `src/api/controller.py`

### Concept: What does a controller do?

The controller sits between the router (HTTP) and the business layer (logic). Its job:
1. Receive validated DTOs from the router
2. Call business functions
3. Handle errors and convert them to HTTP responses
4. Return response DTOs

```python
from fastapi import HTTPException
from src.database.dto import ChatMessageRequest, ChatMessageResponse
from src.business.chatbot import process_chat_message as _process

class ChatController:
    async def send_message(self, request: ChatMessageRequest) -> ChatMessageResponse:
        # Input validation beyond what Pydantic does
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        try:
            result = await _process(request)
            return result
        except Exception as e:
            # Log the real error, return a safe message to the client
            print(f"Error processing chat message: {e}")
            raise HTTPException(status_code=500, detail="Failed to process message")

    async def get_history(self, user_id: int, session_id: str = None, limit: int = 50):
        try:
            from src.business.chatbot import get_chat_history, ChatHistoryRequest
            return await get_chat_history(ChatHistoryRequest(
                user_id=user_id,
                session_id=session_id,
                limit=limit,
            ))
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to retrieve history")
```

### Why not put all logic in the router?

If your route handlers grow to 50+ lines, they become hard to test. Controllers can be instantiated and tested independently of FastAPI:

```python
# In a test
controller = ChatController()
response = await controller.send_message(ChatMessageRequest(message="hello", user_id=1))
assert response.reply  # no HTTP needed
```

---

## 22. Frontend (React UI)

**Files:** `src/ui/src/`

### Core components to understand

The UI is built with React and Tailwind CSS. The key components are:

**`MessageBubble.jsx`** — Renders a single chat message (user or assistant). It handles:
- Different visual styles for user vs. assistant messages
- Markdown rendering for assistant responses
- Timestamps

**`RAGPanel.jsx`** — The right-side panel for the RAG feature. It handles:
- PDF file upload with drag-and-drop
- Sending questions to the RAG endpoint
- Displaying the answer with source citations

### Communicating with the FastAPI backend

```javascript
// POST /api/v1/chat
async function sendMessage(message, userId, sessionId) {
    const response = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, user_id: userId, session_id: sessionId }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail);
    }
    return response.json(); // { reply, session_id, timestamp, ... }
}

// POST /api/v1/rag/upload (multipart form)
async function uploadPDF(file) {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch("/api/v1/rag/upload", {
        method: "POST",
        body: formData,  // no Content-Type header — browser sets it automatically
    });
    return response.json();
}
```

### Key React pattern: Controlled state for chat

```javascript
function ChatApp() {
    const [messages, setMessages] = useState([]);      // displayed messages
    const [sessionId, setSessionId] = useState(null);  // current session
    const [loading, setLoading] = useState(false);

    const handleSend = async (userText) => {
        // Optimistic update — show user message immediately
        setMessages(prev => [...prev, { role: "user", content: userText }]);
        setLoading(true);

        try {
            const response = await sendMessage(userText, userId, sessionId);
            setSessionId(response.session_id);  // capture session on first message
            setMessages(prev => [...prev, { role: "assistant", content: response.reply }]);
        } catch (error) {
            // Show error in UI
        } finally {
            setLoading(false);
        }
    };
}
```

---

## 23. Putting It All Together — The Build Order

Build in this exact order. Each step depends on the previous ones.

```
Phase 1: Foundation
┌─────────────────────────────────────────────────────┐
│  1. config.yml + .env + utils/config.py             │
│  2. database/dto.py (Pydantic models)               │
│  3. memory/chat_history_manager.py (SQLite)         │
└─────────────────────────────────────────────────────┘
                         ↓
Phase 2: AI Core
┌─────────────────────────────────────────────────────┐
│  4. business/core/embedding.py                      │
│  5. business/core/model.py (LLM wrapper)            │
│  6. business/core/prompt_builder.py                 │
└─────────────────────────────────────────────────────┘
                         ↓
Phase 3: Memory
┌─────────────────────────────────────────────────────┐
│  7. memory/vectordb.py (ChromaDB adapter)           │
│  8. memory/redis_memory.py (short-term)             │
│  9. memory/long_term_memory.py (semantic memory)    │
│ 10. memory/responsecache.py (LLM cache)             │
└─────────────────────────────────────────────────────┘
                         ↓
Phase 4: Chatbot Business Logic
┌─────────────────────────────────────────────────────┐
│ 11. business/chatbot/agentic_chatbot.py             │
│ 12. business/chatbot/__init__.py (factory + entry)  │
└─────────────────────────────────────────────────────┘
                         ↓
Phase 5: RAG Pipeline
┌─────────────────────────────────────────────────────┐
│ 13. rag/pdfingest/chunk.py (text chunking)          │
│ 14. rag/pdfingest/unstructure_pdf_digest.py (parse) │
│ 15. rag/vector_store.py (RAG-specific ChromaDB)     │
│ 16. rag/re_ranker/cross_encoder.py                  │
│ 17. rag/re_ranker/re_ranker.py (gating + sorting)   │
│ 18. rag/re_ranker/orchestrator.py (policy)          │
│ 19. rag/index_builder.py (full ingest pipeline)     │
│ 20. rag/retrieval.py (full answer pipeline)         │
│ 21. rag/__init__.py (entry points)                  │
└─────────────────────────────────────────────────────┘
                         ↓
Phase 6: API Layer
┌─────────────────────────────────────────────────────┐
│ 22. api/ratelimiter.py (token bucket)               │
│ 23. api/controller.py (error handling + dispatch)   │
│ 24. api/router.py (HTTP endpoints)                  │
│ 25. api/main.py (FastAPI app + middleware)          │
└─────────────────────────────────────────────────────┘
                         ↓
Phase 7: Frontend
┌─────────────────────────────────────────────────────┐
│ 26. React components (MessageBubble, RAGPanel)      │
│ 27. API integration (fetch calls)                   │
│ 28. State management (useState, session handling)   │
└─────────────────────────────────────────────────────┘
```

### Recommended learning resources

| Topic | Resource |
|-------|---------|
| FastAPI | fastapi.tiangolo.com — official tutorial |
| Pydantic v2 | docs.pydantic.dev |
| OpenAI API | platform.openai.com/docs |
| ChromaDB | docs.trychroma.com |
| Redis | redis.io/docs |
| Unstructured | docs.unstructured.io |
| Cross-encoders | sbert.net/docs/cross_encoder |
| SQLite | sqlite.org/lang.html (SQL reference) |
| React hooks | react.dev/reference/react |

### Concepts you must understand before starting

1. **Python async/await** — FastAPI is async. Understand event loops and when to use `async def` vs `def`.
2. **Pydantic models** — The backbone of type safety in this project.
3. **Vector similarity** — Understand what embeddings are and how cosine similarity works.
4. **OpenAI Chat Completions API** — The `messages` array pattern, roles (system/user/assistant/tool).
5. **HTTP fundamentals** — GET vs POST, status codes (200/201/400/401/422/429/500), headers.
6. **SQL basics** — CREATE TABLE, INSERT, SELECT, WHERE, JOIN, indexes.
7. **Redis data structures** — String, List, and Hash types with TTL.
