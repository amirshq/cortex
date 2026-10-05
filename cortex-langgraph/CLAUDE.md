# CLAUDE.md

Guidance for Claude Code (and other agents) working in `cortex-langgraph/`.

## Project overview

cortex-langgraph is the **LangChain + LangGraph** implementation of Cortex, a
learning-oriented personal-assistant chatbot (FastAPI backend + React/Vite frontend). Its
sibling `../cortex-core/` is the same assistant written by hand, with no agent framework. The
two exist to be **compared**, so keep them feature-equivalent: same API contract, same
prompts, same defaults. When behaviour must differ, document it in README.md under
"Behaviour differences from cortex-core" and pin it with a test.

The codebase doubles as a teaching resource. Module docstrings explain what each LangChain /
LangGraph piece replaced from cortex-core and why. Treat verbose comments as intentional,
and don't strip them during unrelated edits.

Capabilities: an agentic chat (LangGraph tool loop with semantic memory recall + web search),
RAG over PDFs (a LangGraph retrieve → rerank → generate graph), token-bucket rate limiting,
per-model cost tracking, and Prometheus/Grafana/Alertmanager monitoring.

## Environment — use this project's own venv

`.venv/` here is Python 3.11 with the pinned `requirements.txt` (LangChain 1.6, LangGraph 1.2,
openai 3.x). **Do not** install these into the shared/global interpreter: cortex-core pins
`openai==1.3.7`, and LangChain would upgrade it and break cortex-core.

```bash
.venv/bin/python -m pytest                           # hermetic tier
.venv/bin/uvicorn src.api.main:app --reload          # backend on :8000
cd src/ui && npm install && npm run dev              # frontend
cd Docker && docker compose up -d --build            # full stack (same ports/names as cortex-core — run one at a time)
```

## Layout

```
src/
  api/            FastAPI app, router, controllers, rate limiter, metrics — framework-free, unchanged from core.
  business/
    chatbot/
      agentic_chatbot.py  AgenticChatbot: LangGraph StateGraph (agent ⇄ ToolNode via tools_condition),
                          3-way persistence fan-out, token-usage summing, MAX_TOOL_ROUNDS cap.
      tools.py            build_tools(): @tool search_vector_db / web_search, bound per user by closure.
      __init__.py         process_chat_message / get_chat_history — wiring, returns token counts.
    core/
      model.py            create_llm() → BaseChatModel (ChatOpenAI / AzureChatOpenAI / ChatHuggingFace),
                          azure_openai_settings(), model_label(), resolve_provider().
      embedding.py        create_embedder() → Embeddings; optional CacheBackedEmbeddings (cache_dir=).
      prompt_builder.py   build_rag_prompt() ChatPromptTemplate, format_context(),
                          build_agentic_system_prompt() (PromptTemplate).
      cost.py, live_data.py   unchanged from core.
    rag/
      retrieval.py        RAGPipeline: LangGraph StateGraph retrieve → rerank → {generate | fallback | refuse}.
      index_builder.py    build_index(): DoclingLoader → splitter → VectorStore.add_documents (batches of 50).
      vector_store.py     create_vector_store() (Chroma / AzureSearch), reset_vector_store().
      pdfingest/pdf_digest.py  DoclingLoader config (MARKDOWN export, label filter, CortexMetaExtractor).
      pdfingest/chunk.py       split_documents(): RecursiveCharacterTextSplitter, stable sha1 ids, section tags.
      re_ranker/          config.py (ReRankerConfig), cross_encoder.py (CrossEncoderScorer: BaseCrossEncoder),
                          re_ranker.py (rerank_documents: score → gate → sort → truncate).
  memory/
    redis_memory.py       Short-term memory, plain Redis lists + TTL (NOT a LangGraph checkpointer — see below).
    long_term_memory.py   LongTermMemory over any LangChain VectorStore (remember/recall/forget per user).
    vectordb.py           create_conversation_vector_store() → langchain_chroma.Chroma.
    chat_history_manager.py  SQLite sessions/messages — backs /history and /sessions.
  database/dto.py     Pydantic DTOs — the API contract; must stay identical to cortex-core's.
  config/config.yml   Non-secret settings — the ONLY place models are named (models:).
tests/                See "Testing".
```

## Key design decisions (don't undo without reason)

- **Short-term memory is plain Redis, not a checkpointer.** `langgraph-checkpoint-redis` needs
  RedisJSON + RediSearch. The Docker `redis:7-alpine` image, Homebrew Redis and Azure Cache for
  Redis Basic/Standard lack them. Turns go into the graph as LangChain messages via
  `convert_to_messages()`.
- **The relevance gate is ours.** LangChain's `CrossEncoderReranker` drops scores and has no
  threshold. The policy for a fully-gated result is a conditional edge (`_route_after_rerank`).
- **Cross-encoder scores stay raw logits** (`activation_fn=Identity`) for parity with core.
  sentence-transformers would otherwise apply a sigmoid, and `min_score=0.15` would then gate
  different chunks. See the eval finding below.
- **Avoid `langchain-community`.** It's being retired and emits a DeprecationWarning, and
  `pytest.ini` fails the run on deprecations raised from `src.*`. Prefer standalone
  integration packages (`langchain-chroma`, `langchain-azure-ai`, `langchain-docling`) or
  implement the `langchain_core` interface directly.
- **LangGraph's default recursion limit is 10,007**, which is effectively unbounded. The agent
  graph is compiled with `.with_config(recursion_limit=2 * MAX_TOOL_ROUNDS + 2)`.
- **Azure Search index default is `rag-chunks-langchain`**, not core's `rag-chunks`. The
  schemas are incompatible.

## Configuration

`.env` (gitignored) holds secrets and per-environment values; `src/config/config.yml` holds
non-secret settings.

**Models are configured only in `config.yml` → `models:`**, one entry per role: `chat` (the
agent), `rag` (RAG answers, including `temperature` / `max_tokens`), `embedding` and
`reranker`. Never write a model name in code. Read it with `model_settings(role)` from
`src/utils/config.py`, or pass `role=` to `create_llm()`.
- An explicit `model_name=` argument still overrides the config, for tests and evals.
- Azure takes its deployment names from `.env` instead; the Hugging Face provider needs
  `HF_MODEL_NAME`.
- `load_config()` **rejects duplicate YAML keys**. A duplicated `llm_config:` block once
  silently wiped the RAG system role.
- In tests, use the `override_config({...})` fixture to change config values.

Other config sections: `prompts.rag_system_role`, `agent.max_tool_rounds` and `directories`.

Provider switches (all default to the local/on-prem choice):

| Env var | Default | Alternatives |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `azure_openai` · `huggingface` (RAG only — the agent needs tool calling and raises `NotImplementedError`) |
| `EMBEDDING_PROVIDER` | `openai` | `azure_openai` |
| `VECTOR_STORE_PROVIDER` | `chroma` | `azure_search` (Basic tier+; `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, optional `AZURE_SEARCH_INDEX_NAME`, `AZURE_SEARCH_EMBEDDING_DIM`) |
| `CHAT_VECTOR_STORE_PROVIDER` | `chroma` | `azure_search` — not implemented yet |
| `MEMORY_PROVIDER` | `redis` | `azure_redis` (`AZURE_REDIS_CONNECTION_STRING`, `rediss://` only) |
| `LIVE_DATA_PROVIDER` | `mock` | `duckduckgo` · `newsapi` (`NEWS_API_KEY`) |

Azure OpenAI needs `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, optional
`AZURE_OPENAI_API_VERSION`, and `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` /
`AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`. RAG chunks and conversation memory are **two
separate indexes** by design.

## Testing

Two tiers; plain `pytest` runs only the first.

**Tier 1 — hermetic (657 + 10 known-bug xfails, ~20s).** No network, no keys. The shared fakes
in `tests/conftest.py` implement the real LangChain interfaces:
- `FakeChatModel(BaseChatModel)` — replays scripted `AIMessage`s (build them with `reply()`,
  `tool_call()`, `tool_calls()`), records every call in `.calls`, and raises scripted exceptions.
- `FakeEmbedder(Embeddings)` — deterministic vectors; records query and document calls.
- `conversation_store` fixture — a REAL Chroma store in `tmp_path`.
- `FakeRedisMemory` — async in-memory short-term memory.

`tests/api/conftest.py` blocks all non-loopback network access and redirects every data path
into `tmp_path`. Async tests need `@pytest.mark.asyncio` (`asyncio_mode = strict`).
DeprecationWarnings raised from `src.*` fail the run.

**Tier 2 — evals (35, `pytest -m eval -s`).** Real OpenAI, a real cross-encoder, real cost
(cents). They use the fictional golden corpus in `tests/evals/data/golden_set.json`.
`selection_pipeline` runs the real RAG graph with a no-op LLM, so selection is measured for free.

**Known finding pinned by `TestRerankerGate`** (inherited from core on purpose): the
cross-encoder emits raw logits (about ±10), but `min_score=0.15` reads like a 0-1 threshold,
so the effective gate is sigmoid(0.15) ≈ 0.54. One golden query gets gated out and is
recovered by the fallback branch. Fixing the scale should deliberately break those tests.

The xfail tests in `tests/api/e2e/` pin known API bugs shared with cortex-core. They're
`strict=True`, so a fix turns them into failures that prompt removing the marker.

No lint/format tooling is configured; match the surrounding style by hand.

## Conventions

- Keep the router thin: HTTP concerns in `router.py` / `controller.py`, logic in `src/business/`.
  New endpoints use DTOs from `src/database/dto.py`, never raw `dict` bodies.
- New LLM-facing code should use the LangChain interfaces (`BaseChatModel`, `Embeddings`,
  `VectorStore`, `BaseCrossEncoder`) so the fakes in `tests/conftest.py` keep working.
- Prometheus metric names use `_total` / `_seconds` suffixes. Never label a metric with
  high-cardinality values (user IDs, session IDs).

## Known gaps

- `src/database/database.py` and `src/api/service.py` are empty placeholders.
- `src/Learn/` teaching docs describe cortex-core's hand-written internals in places.
- The Alertmanager receiver is a placeholder (no Slack, email or PagerDuty).
- Vector store artifacts under `src/business/rag/vectorstore/` and `data/` are build output,
  not source.
