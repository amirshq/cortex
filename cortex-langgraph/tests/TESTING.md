# cortex-langgraph test suite — what is tested and what is measured

The tests live in three folders, each answering a different question:

```
tests/
  correctness/   Does the code WORK?        pass/fail · fakes · no network · free · ~20s
    conftest.py    shared fakes (FakeChatModel, FakeEmbedder, ...) + the network guard
    api/           controllers, rate limiter, HTTP integration, E2E workflows
    business/      agent, tools, model/embedding factories, prompts, cost, RAG pipeline
    memory/        Redis, long-term memory, SQLite history, caches
    utils/         config loading
  evals/         How GOOD are the results?  numbers · real models · costs cents
    conftest.py    real embedder, real cross-encoder, indexed fictional corpus
    test_*.py      pytest -m eval: retrieval quality, hallucination, latency (fictional corpus)
    eval_golden.py golden-set evaluation on a real 535-page book (a script, not a test)
    data/          golden sets
    results/       saved eval_golden.py runs, one JSON per experiment
  tools/         The same questions with STANDARD TOOLS + a live view · opt-in (pytest -m tools)
    langfuse/      traces, live cost/latency, scores, feedback, datasets
    ragas/         context precision, faithfulness, relevance; test-set generation
    deepeval/      the regression gate
    grafana/       system health, alerts, knowledge-base monitoring
```

`tools/` builds on `evals/` and never replaces it. See [`tools/README.md`](tools/README.md).

| Tier | Count | Cost | Deterministic? | Command |
|---|---|---|---|---|
| 1 — correctness (unit + API integration + API E2E) | 715 (705 pass, 10 known-bug xfails) | free, ~20s | yes | `.venv/bin/python -m pytest` |
| 2 — evaluations | 35 tests + `eval_golden.py` | real API calls, cents | no | `.venv/bin/python -m pytest -m eval -v -s` · `.venv/bin/python tests/evals/eval_golden.py --label <name>` |

`pytest.ini` enforces the split (`addopts = -m "not eval and not tools"`). Two settings matter when you
add a test:

- `asyncio_mode = strict`: every async test needs `@pytest.mark.asyncio`.
- `filterwarnings = error::DeprecationWarning:src.*`: a deprecation raised from our own code
  fails the run. Anything from `langchain-community` would trip this, which is one reason
  the project doesn't use it.

## Evaluation roadmap

The step-by-step plan for evaluating the RAG system, with a status for each step, is in
[`evals/README.md` → The testing roadmap](evals/README.md#the-testing-roadmap):

| Phase | What happens |
|---|---|
| 0 | Foundations: correctness tests, golden set, evaluator, baseline |
| 1 | Settings made configurable |
| 2 | Evaluator extended: re-ranker lift, latency, cost, LLM judge |
| 3 | Experiments: one component at a time |
| 4 | Lock in the winners |

It is kept in that one place so the statuses don't drift apart.

## How the fakes work

Every fake implements the real LangChain interface, so production code can't tell it apart
from the real thing (`tests/correctness/conftest.py`):

| Fake | Interface | Notes |
|---|---|---|
| `FakeChatModel` | `BaseChatModel` | Replays scripted `AIMessage`s: `reply(text, usage=)`, `tool_call(name, args, id)`, `tool_calls([...])`. Records every call's full message list in `.calls` and the bound tools in `.bound_tools`. A scripted exception is raised. Running out of replies fails the test. |
| `FakeEmbedder` | `Embeddings` | Deterministic vectors; records `query_calls` and `document_calls`. |
| `conversation_store` fixture | `VectorStore` | **Real** Chroma, persisted in `tmp_path`, telemetry off. |
| `FakeRedisMemory` | short-term memory | Async, in-memory. |

`tests/correctness/api/conftest.py` adds autouse guards for every API test: dummy credentials, local
providers, a socket guard that fails any non-loopback connection, all data paths redirected
into `tmp_path`, and a fresh rate-limit bucket per test.

## Tier 1 by file

Paths are relative to `tests/correctness/`. Counts in the tables below predate later additions;
run `pytest --collect-only -q` for current numbers.

### Business layer — LangGraph / LangChain code (rewritten for this project)

| File | Tests | What it pins |
|---|---|---|
| `business/chatbot/test_agentic_chatbot.py` | 45 | **Provider guard**: HF → `NotImplementedError`; openai needs a key; the azure deployment becomes the model label. **Tool schema** as the model receives it (`convert_to_openai_tool`), and that both tools are bound. **Tool behaviour**: recall text, "no hits", web-result formatting, contained exceptions, and a hallucinated tool name → error `ToolMessage`, not an exception. **The graph**: nodes, zero/one/multi-hop, parallel calls keyed by `tool_call_id`, tool output fed back to the model, and **`MAX_TOOL_ROUNDS` enforced** (exactly 10 rounds allowed; the 11th raises `GraphRecursionError`). **chat()**: system prompt first, Redis turns replayed as Human/AI messages, prefetched memories and user info in the system prompt, SQLite summary hydration rules. **The 3-way persistence fan-out.** **Token usage** summed across every model call in a turn. |
| `business/core/test_model.py` | 29 | `create_llm()` picks ChatOpenAI / AzureChatOpenAI / ChatHuggingFace; env vs. argument precedence; model-name defaults; sampling is forwarded only when given; retries; `azure_openai_settings()`; `model_label()`. |
| `business/core/test_embedding.py` | 19 | `create_embedder()` selection and errors. **Embedding cache**: repeats embedded once, survives a restart, never shared across models. Regression: the `provider:model` namespace used to crash `LocalFileStore`. |
| `business/core/test_prompt_builder.py` | 34 | RAG `ChatPromptTemplate`: only `{question}` / `{context}` left to fill, system then human, **the grounding rules**, numbered context, braces in PDF text inserted verbatim. Agent prompt: user info, memories, summary, **today's date**, tool instructions, braces in user data inserted verbatim. |
| `business/rag/test_retrieval.py` | 19 | The RAG graph against real Chroma + a scripted cross-encoder + `FakeChatModel`. Checks the graph shape, candidates carry distances and ids, and only re-ranked chunks reach the LLM (sorted, truncated). **Fallback branch** (hybrid / fail_open): vector order, `confidence="low"`. **Refuse branch** (fail_closed): no LLM call at all. Also the default RAG model settings match core. |
| `business/rag/re_ranker/test_re_ranker.py` | 16 | `rerank_documents`: sort, gate, empty, `top_k_input` / `top_n_output`, input not mutated, blending. `CrossEncoderScorer`: it is a `BaseCrossEncoder`, **keeps raw logits (`Identity` activation)**, returns plain floats, uses the batch size. |
| `business/rag/test_ingestion.py` | 41 | **Loader config**: Markdown export; the cleaning label filter (headers, footers, footnotes and captions dropped); OCR on/off by strategy; `CortexMetaExtractor` metadata. **Splitting**: size limit, paragraph-boundary cuts, offsets point back into the source, flat metadata, stable unique sha1 ids, `table` / `text` section tagging. **build_index**: batches of 50, one store connection, upserts by stable id, option forwarding. |
| `business/rag/test_vector_store.py` | 18 | Factory selection; Azure settings, including the dimension passed up front (no paid probe embedding) and the `rag-chunks-langchain` default. **Real Chroma round trip**: distances, ids, upsert, persistence. Reset: empties only its own collection; the Azure reset drops the index and tolerates a missing one. |
| `business/rag/test_rag_entrypoints.py` | 16 | `query_rag` maps the graph's `confidence` to metrics (high → top-score histogram; low / none / empty → low-confidence counter), trims sources to 400 chars and reports the right score, and keeps score bookkeeping out of `metadata`. `ingest_pdfs` resets **before** building. |
| `memory/test_long_term_memory.py` | 28 | remember / remember_conversation / recall / forget_user against **real Chroma**: user scoping, `top_k`, the result shape, unique ids (including within one `remember` call). |
| `memory/test_vectordb.py` | 10 | Conversation-store factory, and **persistence across a new client** (cortex-core's store silently didn't persist). |

### Configuration

| File | Tests | What it pins |
|---|---|---|
| `utils/test_config.py` | 15 | The shipped `config.yml` loads with **no duplicate keys** and configures every model role, the RAG sampling settings, the system role and the tool cap. Duplicate keys are rejected (top-level and nested, with the line number). `model_settings()` gives clear errors and reads the file at call time. |

Each factory's tests also check that **changing only the config changes the model**
(`override_config` fixture), that explicit arguments still win, and that Azure uses its
deployment name rather than the config's model name.

### Unchanged from cortex-core (framework-free code)

| File | Tests |
|---|---|
| `api/test_ratelimiter.py` | 23 |
| `api/test_controller.py`, `test_controller_sessions.py`, `test_controller_branches.py` | 24 · 19 · 12 |
| `api/test_api_test_harness.py` — guards the autouse guards | 6 |
| `api/integration/` — real FastAPI app, business layer mocked (`-m integration`) | 110 |
| `business/core/test_cost.py`, `test_live_data.py` | 26 · 29 |
| `memory/test_redis_memory.py`, `test_chat_history_manager.py`, `test_responsecache.py` | 28 · 35 · 24 |

`test_controller_branches.py` pins the controller's cost-recording branch. In cortex-core
that branch was unreachable; here `process_chat_message` returns real token counts, so it
runs in production.

### API E2E (`-m e2e`) — real app + real business wiring

| File | Tests | Faked |
|---|---|---|
| `api/e2e/test_chat_workflow_e2e.py` | 13 (4 of them xfail) | Only the chat model, the embedding API and Redis. The LangGraph agent, real Chroma memory and real SQLite all run. Covers the session lifecycle, short-term memory reaching the model, the fan-out, cross-session recall through the tool, ownership, and rejected requests leaving no trace. |
| `api/e2e/test_rag_workflow_e2e.py` | 11 (1 of them xfail) | `RAGPipeline` and the index reset + builder. Covers source shaping, the confidence → metrics mapping, upload → reset → rebuild. |

The 5 E2E xfails (`strict=True`) pin known API bugs shared with cortex-core: the echoed
`session_id`, history not scoped to its owner, delete-before-ownership-check, delete not
reaching Redis or Chroma, and a failed upload destroying the previous index.

## Tier 2 — evaluations

Same corpus, metrics and thresholds as cortex-core, so the two implementations can be
compared number for number.

- `evals/test_retrieval_quality.py`: recall@1 / recall@3, MRR, re-ranker lift, gate
  behaviour, embedding sanity. The lift and gate tests run the **real RAG graph** via
  `selection_pipeline` (a no-op LLM, so they're free).
- `evals/test_hallucination.py`: groundedness, refusal rate, LLM-as-judge (with a
  calibration test). `rag_generate()` runs the production `prompt | llm | parser` chain.
- `evals/test_latency.py`: per-stage p50/p95. Stage attribution times the graph's own
  node functions.
- `evals/eval_golden.py`: the same questions asked of a **real document**: 30 questions about
  *AI Engineering* (535 pages). It reports:
  - index coverage, recall@k and MRR;
  - re-ranker lift;
  - latency per stage;
  - cost per question;
  - context recall, accuracy and refusal rate.

  Every run is saved to `evals/results/`. See `evals/README.md`.

The **TestRerankerGate** finding (raw logits vs. a 0-1-looking `min_score`) is inherited on
purpose, because `CrossEncoderScorer` keeps raw logits for parity.

Not yet measured for this implementation: the eval numbers. Run tier 2 and record them here
next to cortex-core's (2026-09-09: recall@1 1.000, MRR 1.000, grounded accuracy 1.000,
refusal rate 1.000, judged groundedness 1.000, RAG p95 2.87s).
