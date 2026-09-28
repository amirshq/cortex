# Cortex test suite — what is tested and what is measured

664 tests in two tiers. `pytest` runs the first tier only.

| Tier | Count | Cost | Deterministic? | Command |
|---|---|---|---|---|
| 1 — default run (unit + API integration + API E2E) | 629 (619 pass, 10 known-bug xfails) | free, ~19s | yes | `pytest` |
| 2 — evaluations | 35 | real API calls, minutes | no | `pytest -m eval -v -s` |

The split is enforced in `pytest.ini` via `addopts = -m "not eval"`. Tier 1 is
hermetic: no network, no API keys, no GPU. Every external service is replaced by
a fake in `tests/conftest.py`.

Two `pytest.ini` settings are worth knowing before you add a test:

- `asyncio_mode = strict` — every async test must carry `@pytest.mark.asyncio`.
- `filterwarnings = error::DeprecationWarning:src.*` — a deprecation raised from
  our own code fails the run instead of scrolling past in the warnings summary.
- `markers` — `integration` and `e2e` select the API's HTTP-level suites
  (`pytest -m integration`, `pytest -m e2e`). Both are hermetic and part of the
  default run; the markers exist for selection, not exclusion.

---

## Tier 1 — default run (629)

### API layer

#### `tests/api/test_ratelimiter.py` — 23 tests

The token-bucket limiter. Grouped into initialisation, consumption, refill,
edge cases, real-world scenarios and concurrency.

**What it asserts:** the bucket starts full; `consume(n)` succeeds only when
`n` tokens are available; tokens refill at `refill_rate` per second and never
exceed `capacity`; fractional tokens accumulate correctly across sub-second
calls; a burst followed by a wait recovers exactly the elapsed allowance.

**Metrics:** none — these are exact-value assertions on `bucket.tokens`. The
limiter's *speed* is measured separately in the latency evals.

#### `tests/api/test_controller.py` — 24 tests

`ChatController.send_message` / `get_history`, `RAGController.query` / `upload`.
`process_chat_message` and `query_rag` are mocked wholesale, so what is under
test is the HTTP boundary, not the business logic.

**What it asserts:** empty/whitespace-only input becomes a 400; a business-layer
exception becomes a 500; an already-raised `HTTPException` is re-raised rather
than swallowed into a 500; Prometheus counters are incremented; PDF upload
rejects non-`.pdf` files, accepts case-variant extensions (`.PDF`), and removes
previously uploaded PDFs before writing the new one.

#### `tests/api/test_controller_sessions.py` — 19 tests

The session list/delete endpoints, which shipped without tests. Same mocking
strategy — `ChatHistoryManager` is mocked, since the SQL itself is covered
against real SQLite in `tests/memory/test_chat_history_manager.py`.

**What it asserts:** `user_id <= 0` and empty/whitespace `session_id` are
rejected *before* the database is touched; an unknown session, or one belonging
to another user, returns 404 (not 403 — no existence leak); storage errors
become 500; validation 400s and 404s are not collapsed into 500s.

#### `tests/api/test_controller_branches.py` — 12 tests

The three controller paths the tests above never executed (branch coverage
found them). Brings `controller.py` to 100% statement and branch coverage.

**What it asserts:** when the business layer reports input/output tokens,
`calculate_chat_cost` is called with the model, both counts and the normalised
`LLM_PROVIDER`; a positive cost lands in `chat_cost_total{model}` and a zero cost
is not recorded; either token count alone triggers costing; with no counts —
the shape `process_chat_message` actually returns today — costing is skipped. A
prebuilt `ChatMessageResponse` from the business layer is returned unchanged. A
4xx `HTTPException` raised during PDF ingestion keeps its status and detail
instead of becoming a 500, and records no indexing metrics.

**Note:** the cost branch is unreachable in production today, because
`process_chat_message` never returns token counts. The tests pin the
controller's side so costing works once the counts are wired through.

#### `tests/api/test_api_test_harness.py` — 6 tests

Guards the guards. If an edit to `tests/api/conftest.py` silently disabled an
autouse fixture, every other test would still pass while reaching the network
or the real `data/` directory. **What it asserts:** outbound connections and
external DNS lookups raise; credentials are dummies; `LIVE_DATA_PROVIDER` and
`SQLITE_DB_PATH` are pinned; `_PROJECT_ROOT` and `ChatHistoryManager` point into
`tmp_path`; each test starts with a full rate-limit bucket.

#### `tests/api/conftest.py` — the API test harness (autouse, applies to every test under `tests/api/`)

- **Dummy credentials, not deleted ones.** `load_dotenv()` never overrides a
  variable that is already set, and `AgenticChatbot.__init__` calls it on every
  chat request — so deleting a key lets `.env` put the real one straight back.
  Every credential is set to `test-dummy-…` instead, and providers are pinned to
  local values (`LIVE_DATA_PROVIDER=mock`).
- **Socket guard.** Non-loopback `connect()` and DNS lookups raise. `TestClient`
  drives the app in-process and opens no sockets, so this only fires on a
  missed mock.
- **Sandboxed paths.** `_PROJECT_ROOT` (controller, chatbot and RAG packages)
  and `ChatHistoryManager` are redirected into `tmp_path`. Upload deletes every
  PDF in `data/rag_uploads`, `ingest_pdfs` resets the RAG index, and the session
  endpoints open the relative path `data/chatbot.db`. **Before this fixture
  existed, the upload tests in `test_controller.py` truncated the real
  `data/rag_uploads/test.pdf` on every run** — they patch `Path.mkdir` but not
  `dest.open("wb")`.
- **Fresh rate-limit bucket per test**, copying the production capacity and
  refill rate, so tests don't depend on run order.
- Opt-in: `client` (`TestClient(app, raise_server_exceptions=False)` so an
  unhandled error returns the 500 a real client would see), `fake_clock`
  (patches the `time` name inside `src.api.ratelimiter` only, never the global
  `time.time`), and `metric` (reads the process-global Prometheus registry;
  assert deltas, never absolute values).

### API integration — `tests/api/integration/` (110 tests, marker `integration`)

Real HTTP through the real app: Prometheus middleware → CORS → router →
rate-limit dependency → controller. Only the business entry points
(`process_chat_message`, `get_chat_history`, `query_rag`, `ingest_pdfs`) and
`ChatHistoryManager` are mocked, in `integration/conftest.py`. Brings `router.py`,
`main.py` and `metrics.py` from 0–53% to 100%.

#### `test_http_routes.py` — 28 tests

Each of the six `/api/v1` operations: body, query and path parameters bind into
the right DTO (including URL-decoded session ids and the int→string `user_id`
conversion storage needs); response bodies match their response models, with
storage-only fields filtered out; unknown body fields are ignored. The OpenAPI
schema lists exactly the six versioned operations; none exist without the
prefix (404); wrong methods are 405; unknown routes are 404.

#### `test_http_validation.py` — 37 tests (3 known-bug xfails)

Which layer rejects what. **FastAPI → 422:** missing, null or wrong-typed fields,
malformed JSON, a JSON array body, a missing `user_id` query parameter, an upload
with no file part — none reach the business layer. **Controller → 400:** blank
message or question, `user_id <= 0`, a blank session id, a non-PDF upload —
rejected before storage is opened or anything is written. **Error mapping:** a
403 from the business layer passes through; an unknown session is 404; a
`ValueError` on the chat path is 400; every endpoint maps an unexpected failure
to 500 with its own detail prefix.

#### `test_http_rate_limit.py` — 14 tests

The limiter as a client sees it, on a fake clock (no sleeps). A request spends
exactly one token through the dependency; the request after capacity gets 429
with `"Rate limit exceeded. Please slow down."` and never reaches the
controller; each rejection increments `rate_limit_rejections_total` and allowed
requests don't; one refill interval restores exactly one request, half an
interval restores none, a long idle refills only to capacity.

Two classes document **current design** rather than a contract (see *Design
notes* below): the bucket is attached only to `POST /chat` and shared by every
caller, and a request FastAPI rejects as 422 has already spent a token.

#### `test_http_metrics_middleware.py` — 10 tests (2 known-bug xfails)

`http_requests_total` increments once with method, path and status;
`http_request_duration_seconds` observes once; `http_requests_in_progress` is
exactly one higher *during* the request (read from inside the handler) and back
to baseline after. Handled 400s, controller 500s and an unhandled exception that
propagates through `call_next` are all counted with the right status, and the
gauge recovers. `/metrics` scrapes are not counted.

#### `test_http_app_shell.py` — 21 tests

`/` and `/health` bodies; `/health` is never rate limited; `/metrics` serves the
Prometheus text format, exposes all 14 Cortex metric families (one parametrised
test each, so a missing family is named), and reflects traffic that just
happened. CORS: a preflight from the Vite dev origin (`http://localhost:5173`)
is allowed for `POST`.

### API end-to-end — `tests/api/e2e/` (24 tests, marker `e2e`)

Real HTTP through the real app **and** the real business wiring. Only external
services are faked (`e2e/conftest.py`): the OpenAI client, the embedder, Redis
and the Chroma conversation store for chat; `RAGPipeline` and the vector store /
index builder behind `ingest_pdfs` for RAG. SQLite is real, in `tmp_path`. The
RAG boundary sits at `RAGPipeline` because the real cross-encoder needs torch, a
model download and ~1s per request; retrieval quality is measured in Tier 2.

#### `test_chat_workflow_e2e.py` — 13 tests (4 known-bug xfails)

**Session lifecycle:** `POST /chat` → `GET /history` shows both messages →
`GET /sessions` lists the session titled with the first message → `DELETE` →
both are empty. Follow-up turns join the session, keep its title, and the earlier
turn is sent to the model. **Memory fan-out:** one turn lands in Redis, the
vector store and SQLite; a second session recalls it through the agent's
`search_vector_db` tool, and the tool result reaches the model. **Ownership:**
another user's delete is 404 and the session is still listed; sessions are
listed per user. **Rejected requests leave no trace:** a 429, a 422, a 400 and a
model failure (500) each reach neither storage nor, where applicable, the model.

#### `test_rag_workflow_e2e.py` — 11 tests (1 known-bug xfail)

**Query:** a reranked answer returns sources trimmed to 400 characters and
scored by `rerank_score`, and records `rag_queries_total` +
`rag_retrieval_top_score`; a gated fallback is scored by vector distance and
counted as low confidence; no context still answers and counts as low
confidence; the pipeline opens its index inside the sandbox; a pipeline failure
is a 500 and not counted as a query. **Upload:** a valid PDF is saved, then the
index is reset and rebuilt from only that PDF, and the indexing counters move; a
second upload indexes only the new file; a rejected type or empty filename
touches neither disk nor index; an indexing failure is a 500 with no indexing
metrics.

### Chatbot orchestration

#### `tests/business/chatbot/test_agentic_chatbot.py` — 38 tests

The largest single file, and the only coverage of the tool-calling loop. The
OpenAI client, Redis, Chroma and SQLite are all replaced by
`tests/conftest.py` fakes, and `FakeCompletions` replays a *scripted* list of
assistant messages so a multi-turn agent trajectory is fully deterministic.

Five groups:

- **Provider selection** — `LLM_PROVIDER=huggingface` raises `NotImplementedError`
  (it used to silently use OpenAI); an unknown provider raises; OpenAI requires
  an API key; Azure requires a deployment name and goes through the shared
  `build_azure_openai_client()`.
- **Tool schema** — both tools (`search_vector_db`, `web_search`) are exposed and
  each declares a required `query` parameter.
- **Tool dispatch** — recalled text is returned; a no-hit recall says so rather
  than returning an empty string; web results are formatted with source and URL,
  and omit the URL line when absent; a provider exception is contained rather
  than propagated; an unknown tool name returns a marker instead of raising.
- **The ReAct loop** — returns immediately when the model requests no tools;
  feeds each tool result back into the message list; handles *parallel* tool
  calls in one assistant message; handles multi-hop (tool → tool → answer);
  sends `tools` and `tool_choice="auto"` on every call. One test pins that the
  loop has **no iteration cap** — deliberate documentation of current behaviour.
- **Context assembly and the 3-way persistence fan-out** — system prompt first
  and user message last; recent Redis turns replayed in between; prefetched
  memories and user info land in the system prompt; a cold session hydrates its
  summary from SQLite and a warm one does not; every turn is written to Redis
  *and* the conversation vector store *and* SQLite; the first turn titles the
  session from the message (truncated to 60 chars) and later turns do not
  re-title; SQLite is optional; a tool-using turn persists only the final answer.

### Core

#### `tests/business/core/test_model.py` — 26 tests

`BaseLLM` / `OpenAIModel` / `LocalHFModel` / `create_llm()` /
`build_azure_openai_client()`.

**What it asserts:** the ABC cannot be instantiated and a subclass must
implement `generate`; the OpenAI client is called with the right model,
temperature and message shape; responses are `.strip()`ped; empty and
multi-item context both work. The factory defaults to `openai`, reads
`LLM_PROVIDER` from the environment, is case-insensitive, and raises a clear
error for an unknown provider or a missing credential. `LocalHFModel` sets
`pad_token` when the tokenizer lacks one.

#### `tests/business/core/test_embedding.py` — 25 tests

`OpenAIEmbedder` and `create_embedder()`.

**What it asserts:** `embed_query` returns a flat list and `embed_documents`
batches; an empty list embeds to an empty list; the backward-compatible
`embed()` delegates to `embed_query`. **Retry logic** gets its own group: a 500
is retried, the attempt count is capped, and the delay grows exponentially
between attempts. Factory tests mirror the LLM factory's.

#### `tests/business/core/test_cost.py` — 26 tests

`calculate_chat_cost` / `calculate_embedding_cost` and the pricing tables.

**Metric under test: USD cost.** `cost = (input_tokens / 1M × input_price) +
(output_tokens / 1M × output_price)`.

**What it asserts:** correct cost for GPT-4o, GPT-3.5-turbo and the Azure
pricing table; partial (sub-million) token counts scale linearly; zero input or
zero output contributes zero; floating-point precision holds at both small and
large token counts; an **unknown model returns $0 rather than raising** — a new
model name must never break a chat request, so the cost is silently zero and the
gap shows up as a flat line in Grafana rather than a 500.

#### `tests/business/core/test_live_data.py` — 29 tests

`MockLiveDataProvider`, `DuckDuckGoSearchProvider`, `NewsAPIProvider`, and
`create_live_data_provider()`.

**What it asserts:** each provider returns the same dict shape so they are
substitutable; `limit` is respected; a genuine no-results search returns `[]`.
Error handling is asserted to be **non-empty and self-describing** rather than
empty: both a raised exception and an API error *response* (HTTP 200 carrying
`status: error`) produce a synthetic result whose `summary` contains "error" or
"failed". That distinction is the point — "the search failed" and "the search
found nothing" reach the LLM as different facts, and a failed search must
degrade the answer, not the request. NewsAPI requires a key and prefers an
explicit parameter over the env var; DuckDuckGo checks the `ddgs` import. The
factory defaults to `mock` (safe for on-prem, no external calls).

#### `tests/business/core/test_prompt_builder.py` — 35 tests

Both prompt builders. The rationale in the file's docstring is the point: a
dropped instruction in a prompt doesn't raise — it just makes the model worse in
a way no other test notices.

**What it asserts:**
- **RAG `PromptBuilder`** — the question and every context chunk appear;
  chunks are numbered from 1; the numbering matches between the text and
  messages variants; `build_messages()` returns system-then-user; the
  **grounding rules survive even with empty context** (that is the path where
  hallucination is most likely).
- **Agentic prompt** — user info renders as key/value lines with a placeholder
  when absent; recalled snippets are included and numbered; the summary is
  included and stripped. **Date grounding**: today's date and the current year
  are stated, and the model is told its training data is older. **Tool
  instructions**: recall before answering, web-search for current information,
  *don't* append a year to search queries, trust search over training data,
  admit uncertainty.

### RAG

#### `tests/business/rag/test_ingestion.py` — 31 tests

The `Chunker`, chunk-id generation, table tagging and `build_index`. Docling
itself is not exercised — that would test a third-party PDF parser.

**What it asserts:** `overlap` must be smaller than `chunk_size`; empty and
whitespace-only text yield no chunks; consecutive chunks overlap by exactly the
configured amount; the chunks **cover the whole document** with no gap; the loop
terminates on text shorter than the overlap (the classic infinite-loop bug).
Chunk ids are a **SHA-1 hex digest, deterministic** for the same
(source, offset) and unique within a document — that is what makes `upsert`
idempotent across re-ingestion. Table tagging: chunks before the table marker
are `section=text`, at or after it are `section=table`, and offsets survive
enrichment. `build_index` embeds in **batches of 50** (an OOM guard), upserts
once per batch with four aligned lists, and connects to the store lazily.

#### `tests/business/rag/test_vector_store.py` — 33 tests

Chroma and Azure AI Search behind one interface, plus the factory. The Azure SDK
is imported lazily inside the class, so these tests inject fake SDK modules into
`sys.modules` rather than requiring `azure-search-documents` to be installed.

**The load-bearing group is `TestAzureToChromaShapeTranslation`.**
`retrieval.py::_retrieve()` indexes `result["ids"][0]`, `["documents"][0]`,
`["metadatas"][0]`, `["distances"][0]` regardless of backend. If Azure's
`query()` stops matching that nested-list shape, RAG breaks **on Azure only, and
silently** — returning zero chunks instead of raising. So these tests assert the
four keys are present, every value is wrapped in an outer batch list, an actual
`_retrieve()` can unpack the Azure response, an empty result set yields empty
*inner* lists, and Azure's similarity score (higher = better) is **negated** into
a distance (lower = better) to match Chroma's convention.

Also asserted: Chroma is created with `embedding_function=None` (we supply
embeddings); `upsert` rejects mismatched list lengths; `reset()` deletes then
recreates, and tolerates a missing index on Azure.

#### `tests/business/rag/test_retrieval.py` — 14 tests

`RAGPipeline._retrieve` and `.answer`. Every constructor collaborator is patched
so no transformer model is downloaded and no API key is needed.

**What it asserts:** the query is embedded exactly once; one `RetrievedChunk` per
hit with every field mapped; missing response keys don't raise; a `None` distance
becomes `0.0` and `None` metadata becomes `{}`. On `answer()`: retrieval uses
`top_k_input`; the LLM receives **re-ranked** text, not raw vector order;
`policy="hybrid"` is passed to the orchestrator; the query reaches the re-ranker
verbatim; empty context still calls the LLM instead of raising.

#### `tests/business/rag/test_rag_entrypoints.py` — 14 tests

`query_rag` / `ingest_pdfs` — what the controller actually calls. These own the
**retrieval-quality metric emission**, which nothing else covers:

| Behaviour | Metric |
|---|---|
| every query | `rag_queries_total` incremented |
| high confidence (a `ReRankedChunk` at position 0) | `rag_retrieval_top_score` observes the rerank score; low-confidence counter **not** touched |
| fallback (a plain `RetrievedChunk` at position 0) | `rag_retrieval_low_confidence_total` incremented; histogram **not** observed |
| no chunks at all | counts as low confidence |

Also: source text is truncated to 400 chars before going over the wire; a
re-ranked source reports `rerank_score` while a fallback source falls through to
`vector_score` via `getattr` rather than raising. And the **reset-before-reindex
rule** — `upsert` never deletes, so re-uploading without a reset leaves stale
chunks from the previous PDF answering questions about the new one; one test
asserts `reset()` is called, another asserts it happens *before* `build_index`
(reversed, it would wipe the new index).

#### `tests/business/rag/re_ranker/test_re_ranker.py` — 3 tests

The scoring maths, with a deterministic `MockScorer` (score = `i * 0.1`).
Asserts output is sorted by `rerank_score` descending, truncated to
`top_n_output`, gated by `min_score`, and that empty input returns empty.

#### `tests/business/rag/test_orchestrator.py` — 2 tests

The `select_context` policies, using an `EmptyScorer` that gates everything out:

- `fail_closed` → `([], "none")` — refuse rather than answer ungrounded.
- `fail_open` / `hybrid` → top-k vector results, `confidence="low"`.

### Memory

#### `tests/memory/test_redis_memory.py` — 28 tests

`RedisMemory` against a faked async client that records
`rpush`/`expire`/`lrange`/`delete`.

**What it asserts:** the `chat:`-prefixed key scheme; messages stored as JSON
with role and content; `rpush` appends rather than replaces; **the TTL is
refreshed on every write** (otherwise an active conversation expires mid-chat);
unicode survives the JSON round-trip; `get_messages` uses negative-index window
arithmetic to read the last N and preserves stored order. Factory tests cover
`azure_redis`: it reuses `RedisMemory` unchanged, requires
`AZURE_REDIS_CONNECTION_STRING`, and **rejects a plaintext `redis://` scheme**
(Azure requires TLS).

#### `tests/memory/test_long_term_memory.py` — 26 tests

The semantic-recall layer. `LongTermMemory` owns memory *semantics*, not
storage, so its collaborators are the in-memory fakes and the assertions are on
the rows that would have been written.

**What it asserts:** `remember()` chunks first, embeds each chunk, and attaches
the documented metadata (`user_id`, `type`, `importance`, ISO-parseable
`created_at`); `remember_conversation()` stores the user/assistant pair as **one**
row in role-prefixed form and does *not* invoke the chunker; `recall()` embeds
the query, **filters by `user_id`** (cross-user leakage is the failure mode
here), defaults to `top_k=5`, and returns the `text`/`metadata`/`score` shape its
callers read; `forget_user()` removes only that user's rows. Ids are
user-prefixed and unique across calls.

#### `tests/memory/test_chat_history_manager.py` — 35 tests

Run against a **real SQLite file** in `tmp_path` — SQLite needs no server, and
mocking would test nothing since the class is entirely SQL.

**What it asserts:** the schema is created (both tables, parent directories) and
`__init__` is idempotent; `ensure_session` is idempotent; `list_sessions` is
scoped to the user, ordered newest-first, and returns the documented keys.
`delete_session` deletes the session **and its messages**, returns `False` for an
unknown session or one belonging to another user, and leaves other sessions
alone — one test in this group pins a **known defect**, see
[Known defects](#known-defects-pinned-by-tests). Messages come back oldest-first
with working `limit`/`offset` pagination;
unicode and newlines round-trip; **SQL metacharacters are parameterised, not
interpolated** (injection). `get_latest_summary` reads only the most recent
session, takes the last N messages oldest-to-newest, and is user-scoped.

#### `tests/memory/test_responsecache.py` — 24 tests

The Redis-backed LLM response cache — which is almost entirely a
key-derivation problem with two expensive failure modes: keys too coarse serves
one user another user's answer (or a GPT-3.5 answer for a GPT-4o request), keys
too fine means nothing ever hits and the cache costs money instead of saving it.

**What it asserts:** the hash is a SHA-256 hex digest; identical payloads hash
identically and **dict key order does not matter** (so an equivalent request
still hits); whitespace differences *do* change the hash; nested payloads are
hashable. The key layout embeds user and model, so different users and different
models get different keys. `get`/`set` use the same derived key, an empty cached
string is treated as a miss, the TTL is applied on write (default 15 min), and
`invalidate_user` scans with a user-scoped pattern and deletes every match.

#### `tests/memory/test_vectordb.py` — 22 tests

`ChromaVectorDB`, the conversation-memory store. Note the response translation
here is the **opposite** of the RAG store's: Chroma's nested lists are
*flattened* into a list of dicts, because `LongTermMemory.recall()`'s callers
iterate results and read `r["text"]`.

**What it asserts:** `add` forwards all four lists and supports batches; `search`
flattens correctly, preserves Chroma's ordering, wraps the embedding in a batch
list, defaults to `top_k=5`, and passes filters through as `where` (or `None`);
`delete` deletes by filter. The factory tests assert
`CHAT_VECTOR_STORE_PROVIDER` is **independent of** the RAG switch, and that
`azure_search` raises `NotImplementedError` rather than silently falling back.

---

## Tier 2 — evaluations (35, marked `eval`)

```bash
pytest -m eval -v -s      # -s matters: each test prints its measured metric
```

These use real models and real money, and they answer questions unit tests
structurally cannot: *is the right chunk retrieved, does the system refuse what
it doesn't know, and where does the latency go.*

### The corpus

`tests/evals/data/golden_set.json`: 5 documents, 6 retrieval cases, 4 grounded
cases, 4 unanswerable cases. It describes a **fictional** company (Veldrin Corp,
Kestrel-7 sensor) on purpose — if the model can answer without retrieval, it is
fabricating, because no such facts exist in any training set. A corpus of real
facts cannot tell retrieval apart from memorisation.

One document contains generic humidity theory and is named
`distractor-weather`: it is on-topic and answers nothing, which is exactly what
semantic search gets fooled by.

Extend the JSON, not the test code. Every fixture in `tests/evals/conftest.py`
**skips** rather than fails when its prerequisite is missing, so a run without
`OPENAI_API_KEY` says "skipped", not "broken".

### `tests/evals/test_retrieval_quality.py` — 14 tests

#### recall@k

```
recall@k = (1/N) · Σ  1 if any relevant doc appears in the top k, else 0
```

Per query it is binary. Each golden case has exactly one relevant document, so
this is a hit-rate: "did the answer-bearing chunk make the cut".

- **recall@1** — floor `0.66`. If the top hit is usually wrong, nothing
  downstream recovers: the re-ranker only reorders what retrieval returned.
- **recall@3** — floor `0.95`. This is the number that matters for answer
  quality, because `top_n_output` is 8 — a relevant chunk anywhere in the top few
  still reaches the prompt.

#### MRR (Mean Reciprocal Rank)

```
MRR = (1/N) · Σ 1/rank_of_first_relevant_doc      (0 if none retrieved)
```

Floor `0.75`. Unlike recall@k this is **rank-sensitive**: rank 1 scores 1.0,
rank 2 scores 0.5, rank 3 scores 0.33. It is the metric that notices retrieval
degrading from "right answer first" to "right answer third" — a change recall@3
cannot see at all.

#### Re-ranker lift

```
lift = MRR(via select_context) − MRR(vector-only)
```

Floor `0.0` — i.e. the full path must not be *worse* than raw vector order. The
measurement deliberately goes through `select_context(policy="hybrid")`, not
`ReRanker.re_rank()` directly, because production never calls the latter.
Measuring `re_rank()` alone would report failures the application recovers from,
and would hide that the recovery is happening.

#### Low-confidence fallback rate

```
rate = (queries where the gate rejected every candidate) / N
```

Ceiling `0.35`, currently ~`0.17` (1 of 6). Each occurrence is a query where the
cross-encoder scored everything below `min_score` and the pipeline reverted to
raw vector order — in production, one increment of
`rag_retrieval_low_confidence_total`. A rising count is the early warning that
retrieval is degrading.

#### Diagnostics and pinned findings

- **Per-case breakdown** — prints `ok`/`weak`/`MISS` and the reciprocal rank per
  question. The aggregates say something regressed; this says *what*.
- **Distractor test** — `distractor-weather` must not outrank the spec sheet.
- **`TestRerankerGate`** — pins the known scale bug (see below): scores are
  logits not probabilities; a relevant chunk *can* be gated out entirely; the
  hybrid fallback recovers it and reports `confidence="low"`.
- **`TestEmbeddingConsistency`** — query and document embeddings share
  dimensionality (a mismatch makes every search return noise); the same text
  embeds **stably** — asserted as cosine ≥ 0.9999 rather than bit-equality,
  because OpenAI's endpoint drifts ~6e-5 per component between identical calls;
  and related text scores closer than unrelated text, which is the floor
  assumption under all of RAG.

### `tests/evals/test_hallucination.py` — 10 tests

The failure mode this file exists for: a fluent, confident, well-formatted
answer supported by nothing in the corpus. It has the same type, the same shape
and the same HTTP status as a correct one, so no other test can catch it.

#### Grounded accuracy

```
accuracy = (answers containing the expected fact) / (answerable questions)
```

Floor `0.75`. Keyword matching against each case's `must_contain_any`.

#### Refusal rate

```
refusal rate = (unanswerable questions declined) / (unanswerable questions)
```

Floor `0.75`, deliberately high: confidently inventing a warranty term or a
revenue figure is worse than being useless, because the user cannot tell.
Detection is a keyword list (`REFUSAL_MARKERS` — "I don't know", "not in the
context", "not specified", …).

#### Over-refusal (the opposite failure)

A system that says "I don't know" to everything passes every hallucination test
and is worthless. So a separate test asserts **zero** answerable questions are
refused. The two metrics are a pair; neither is meaningful alone.

#### Judged groundedness (LLM-as-judge)

```
groundedness = (answers judged GROUNDED) / (answerable questions)
```

Floor `0.75`. A second `gpt-4o-mini` call, temperature 0, is given the retrieved
CONTEXT and the ANSWER and replies `GROUNDED` / `UNGROUNDED`. This catches what
keyword matching cannot: an answer that contains the right number *surrounded by
invented detail*. A refusal counts as GROUNDED.

**The judge is itself calibrated.** One test feeds it a correct answer (must say
GROUNDED — otherwise it is too strict to trust) and a planted hallucination
("…and was acquired by Siemens in 2025 for €1.2 billion") which it must reject.
Without this, a judge that answered GROUNDED to everything would make the
groundedness test pass unconditionally and mean nothing.

#### Specific traps

- **Nonexistent product** — there is a Kestrel-7; there is no Kestrel-9. A model
  that answers about one is pattern-matching.
- **Year extrapolation** — financials stop at FY2024; a fluent trend
  extrapolation to FY2026 reads exactly like a retrieved fact.
- **Empty context** — with zero chunks the model must fall back on the prompt's
  "I don't know" rule instead of its training data.
- **Irrelevant context** — given only the humidity-theory distractor, it must not
  stretch it into an answer about the company.
- **Sources always returned** — an answer without sources cannot be verified,
  which is the only defence left once the model is fluent.

### `tests/evals/test_latency.py` — 11 tests

Answers "where does the time actually go", per stage, so a slow endpoint can be
attributed rather than guessed at. Every test prints `n / min / p50 / p95 / max`.

Percentiles use **nearest-rank, no interpolation** (`conftest.percentile`) —
the samples are small, so interpolating would invent precision.

| Stage | p95 budget | What breaking it means |
|---|---|---|
| `embed_query` | 2.0s | network path, or a switch to a larger embedding model |
| vector search (embedding excluded) | 1.0s | a second means Chroma is doing a linear scan, not using its HNSW graph |
| rerank, ~5 candidates on CPU | 10.0s | usually the largest non-LLM cost; regressed by moving off GPU or enlarging `top_k_input` |
| full `answer()` | 30.0s | retrieve + rerank + generate end to end |

Budgets are loose on purpose: they catch a stage getting an *order of magnitude*
slower, not ordinary run-to-run variance.

Also measured:

- **Batching win** — `embed_documents(all)` must beat serial per-text calls, with
  the speedup printed. `index_builder` batches by 50 for memory reasons; this
  confirms it is also a throughput win, not just an OOM guard.
- **Sub-linear scaling in k** — `k=30` must not cost more than `10× k=1 + 0.5s`.
  Sub-linear scaling in k is the entire point of an ANN index.
- **Cold-start cost** — first vs. second `re_rank()` call, with the warm-up
  delta printed. In a fresh container that cost lands on a real user's request.
- **Stage attribution** — one representative query split into
  retrieve / rerank / generate with percentages. Diagnostic, not a gate.

Three latency tests are **hermetic** (CPU only, no network) and therefore stable
enough to gate on:

| Test | Budget |
|---|---|
| chunking throughput | < 5.0 s/MB |
| `TokenBucket.consume()` | < 100 µs/call over 100k calls — it runs on the hot path of every request |
| agentic prompt building | 1000 builds < 5.0s |

### Thresholds are regression floors, not targets

Every threshold is a module-level constant set *below* measured performance,
with a comment saying what breaking it would mean in production. Ordinary model
drift should not turn CI red; a real collapse should. Tighten them as the system
improves.

**Measured 2026-09-09:** recall@1 1.000, MRR 1.000, grounded accuracy 1.000,
refusal rate 1.000, judged groundedness 1.000, end-to-end RAG p95 2.87s
(generate 63%, retrieve 18%, rerank 18%).

---

## Metrics glossary

### Retrieval metrics

| Metric | Formula | Answers |
|---|---|---|
| recall@k | mean over queries of `1 if any relevant doc in top k else 0` | did the answer-bearing chunk make the cut? |
| MRR | mean of `1/rank_of_first_relevant` | how *high* did it rank? |
| re-ranker lift | `MRR_after − MRR_before` | does the cross-encoder earn its latency? |
| low-confidence rate | share of queries falling back to vector order | is the relevance gate rejecting everything? |

### Generation metrics

| Metric | Formula | Answers |
|---|---|---|
| grounded accuracy | share of answerable questions whose answer contains the expected fact | does it answer correctly? |
| refusal rate | share of unanswerable questions declined | does it refuse to invent? |
| over-refusal | share of *answerable* questions wrongly declined | is it uselessly cautious? |
| judged groundedness | share of answers a judge model rules entailed by their sources | is every claim supported, not just the key number? |

### Latency metrics

p50 / p95 per stage, nearest-rank, plus percentage attribution across
retrieve / rerank / generate.

### On "context precision" and "context recall"

These are the RAGAS-style names for the two halves of retrieval quality, and it
is worth being precise about which one this suite measures.

**Context recall** — *of everything needed to answer the question, how much did
retrieval actually bring back?*

```
context recall = (ground-truth claims present in the retrieved context)
                 / (ground-truth claims needed to answer)
```

It is the ceiling on answer quality. A fact that was never retrieved cannot be
in the answer, and no prompt, re-ranker or larger model recovers it. **This suite
measures it as `recall@1` / `recall@3`** — a per-document proxy rather than a
per-claim ratio. Because each golden case declares exactly one
`relevant_doc_ids` entry, recall@k here collapses to a binary hit-rate. That is
adequate for a single-fact corpus and would need to become a real ratio if a
question ever required combining two documents.

**Context precision** — *of what retrieval brought back, how much was actually
relevant, and was the relevant part ranked first?*

```
context precision@k = (relevant chunks in top k) / k
```

It is the noise measure. Low precision doesn't make the answer impossible, it
makes it expensive (tokens paid for irrelevant chunks) and more likely to drift,
because the model has plausible-looking but useless text in its context window.

**Precision is not measured as a ratio in this suite.** The closest coverage,
and what each piece does and does not tell you:

| Test | Covers | Doesn't cover |
|---|---|---|
| `test_mean_reciprocal_rank` | rank-sensitivity — the relevant chunk being *first* is what precision-at-low-k rewards | how much noise sits alongside it |
| `test_distractor_does_not_outrank_the_answer` | one adversarial ordering case | the general ratio |
| `test_gate_filters_the_irrelevant_distractor` | asserts `len(selected) < 5`, i.e. the `min_score` gate discarded *something* | how many of the survivors are relevant |
| low-confidence fallback rate | when gating fails completely | partial gating quality |

To add it properly: label each golden query's *irrelevant* doc ids alongside
`relevant_doc_ids`, then compute `(relevant in top k) / k` averaged over
queries, and separately `precision` over the set that survived the `min_score`
gate — the second number is the one that tells you whether the gate is tuned,
which is precisely what the finding below makes currently unanswerable.

---

## Known defects pinned by tests

### API known bugs — strict xfail (10 tests)

These tests assert the **correct** behaviour, so they fail today and are marked
`xfail(strict=True)`. They show as `x` in a normal run. When a bug is fixed its
test XPASSes, `strict` turns that into a failure, and the marker should be
removed. Each was confirmed with `--runxfail` to fail on its own assertion, not
on a fixture error.

| Bug | Test | Measured today |
|---|---|---|
| Out-of-range `limit` / `offset` on `GET /history` returns 500, not 422 — the router builds `ChatHistoryRequest` inside the handler | `integration/test_http_validation.py::TestKnownIssueHistoryPaginationRange` (3) | `500 == 422` |
| The middleware labels metrics by raw path: every session id and every unmatched URL becomes a new time series | `integration/test_http_metrics_middleware.py::TestKnownIssuePathLabelCardinality` (2) | 21 samples carry the probe id |
| `POST /chat` without `session_id` returns `session_id: null`, though the turn was stored under `str(user_id)` | `e2e/test_chat_workflow_e2e.py::TestKnownIssues::test_response_names_the_session_the_server_used` | `None == '7'` |
| `GET /history` filters by session id only — any `user_id` reads any session | `…::test_history_is_scoped_to_the_session_owner` | user 8 read 2 of user 7's messages |
| Another user's rejected `DELETE` (404) still wipes the owner's messages — defect 1 below, now caught over HTTP | `…::test_a_rejected_delete_by_another_user_leaves_the_owners_messages` | owner's history total `0` |
| `DELETE /sessions` removes SQLite rows only; the turns stay in Redis and the conversation vector store and remain recallable | `…::test_deleting_a_session_removes_its_turns_from_every_memory_tier` | session still in Redis |
| A failed upload has already deleted the previous PDFs and reset the index, leaving nothing indexed | `e2e/test_rag_workflow_e2e.py::TestKnownIssueFailedUploadDestroysExistingContent` | index `[]` |

### Design notes (current behaviour, documented, not asserted as correct)

- **Malformed requests spend rate-limit tokens.** FastAPI resolves the
  `_check_rate_limit` dependency before validating the body, so ten malformed
  `POST /chat` requests (422) exhaust the bucket and the next valid request gets
  429.
- **One bucket, one route.** The limiter is a single module-level bucket shared
  by every caller, attached only to `POST /chat`. `/rag/query` also calls the LLM
  and is unlimited.

### Older pinned defects

Two tests assert **current wrong behaviour on purpose**, so the defect is
visible in CI rather than discovered in production. Both say so in their
docstring, and both should be rewritten when the underlying code is fixed.

### 1. `delete_session()` wipes messages before checking ownership

`tests/memory/test_chat_history_manager.py::test_wrong_user_still_deletes_the_messages`

`delete_session(session_id, user_id)` deletes from the `messages` table *before*
verifying that the session belongs to `user_id`. So a mismatched user gets
`False` back (correct) after the transcript has already been wiped (wrong) — the
session row survives with an empty transcript. The controller maps that `False`
to a 404, so the caller is told nothing happened.

The test asserts `get_messages("s1") == []` — i.e. it pins the data loss. The
fix is to move the message `DELETE` after the session `DELETE` and make it
conditional on rowcount, or to wrap both in one ownership-scoped transaction.
When that lands, the test should be rewritten to assert the messages **survive**.

### 2. The re-ranker gate is applied to the wrong scale

`tests/evals/test_retrieval_quality.py::TestRerankerGate` (3 tests)

`CrossEncoderReRanker._batch_score()` returns the model's **raw logits** —
unbounded, roughly ±10 on this corpus. But `ReRankerConfig.min_score` defaults to
`0.15` and is documented as a relevance threshold, a value that only reads as
sensible on a 0-1 probability scale.

Applying `0.15` to a logit means the effective gate is `sigmoid(0.15) ≈ 0.54`
— "at least 54% relevance probability", roughly 3.5× stricter than the config's
own comment implies.

Measured consequence: *"What radio frequency does the sensor use in Europe?"*
retrieves the correct spec sheet at **vector rank 1**, the cross-encoder scores
it about **−4.2**, the gate drops **every** candidate, and `re_rank()` returns
`[]`. The answer is in the corpus; the gate simply does not believe it.

The hybrid fallback recovers the chunk and marks the query low-confidence, so
answers stay correct — but precision gating has degraded to all-or-nothing: for
an affected query the system silently reverts to unranked vector order.

The three tests in that class **document current behaviour rather than assert a
fix**. If the scoring is changed to emit sigmoid probabilities, or `min_score` is
retuned for the logit scale, they should fail — and should then be deleted and
`MAX_LOW_CONFIDENCE_RATE` tightened.

---

## Fakes and fixtures (`tests/conftest.py`)

Nothing here talks to a real service.

| Fake | Stands in for | Notable property |
|---|---|---|
| `FakeEmbedder` | `OpenAIEmbedder` | vector derived from `sum(ord(c))`, so identical text embeds identically — which is what the id-stability and caching assertions rely on |
| `FakeConversationVectorStore` | `ChromaVectorDB` | in-memory rows, honours a `user_id` filter |
| `FakeRedisMemory` | `RedisMemory` | async, in-memory dict, supports preloaded history |
| `FakeOpenAIClient` / `FakeCompletions` | the OpenAI chat client | replays a **scripted** list of assistant messages one per `create()` call, and records every `messages` list it was handed — so tests can assert on what the agent actually sent. Running out of scripted responses raises a named `AssertionError`, so "the loop called the model more times than expected" is a legible failure |

`conftest.py` also centralises the `sys.path` bootstrap that older test files did
by hand, and provides `tmp_db_path` for the SQLite tests.

The API tests add their own harness in `tests/api/conftest.py` (dummy
credentials, socket guard, sandboxed paths, fresh rate-limit bucket, `client`,
`fake_clock`, `metric`) — described under *Tier 1 → API layer*.

---

## Not tested, on purpose

- **Docling** — a third-party PDF parser. Testing it would test their code.
  Everything built on top of it (chunk arithmetic, ids, table tagging) is tested.
- **`src/database/database.py`** — empty; SQLAlchemy setup is not implemented.
  Chat history persistence goes through `src/memory/` instead.
- **`src/api/service.py`** — empty placeholder.
- **`tests/business/rag/re_ranker/run_reranker_smoke_test.py`** — a manual script
  (no `test_` prefix, not collected). It prints ranked chunks for eyeballing.
