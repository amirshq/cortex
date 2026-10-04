# Cortex Beginner — RAG from scratch

The smallest complete RAG (Retrieval-Augmented Generation) system: about 300 lines of
plain Python, one file per step, no framework. It's the first project in the Cortex series:

| Project | What it adds |
|---|---|
| **cortex-beginner** ← you are here | The five core RAG steps, nothing else |
| [cortex-core](../cortex-core/) | A full app: API, chat agent, memory, PDF/table parsing, monitoring, tests |
| [cortex-langgraph](../cortex-langgraph/) | cortex-core rebuilt with LangChain + LangGraph |

## How RAG works

```
INDEX (once)
  data/*.txt ──► chunks ──► embedding model ──► vectors ──► vector database
                 loader.py   embedder.py                    vector_db.py

ASK (every question)
  question ──► embed ──► vector DB: top 10 ──► re-ranker: best 3 ──► LLM ──► answer
               └──── retriever.py ─────┘       reranker.py           llm.py
```

An LLM only knows what it was trained on. RAG gives it **your** documents at question time:
find the few chunks that answer the question, then ask the LLM to answer from those chunks
only.

## The five building blocks

| Step | File | What it does |
|---|---|---|
| 1. Embedding model | `rag/embedder.py` | Turns text into a vector of 1536 numbers that captures its *meaning* (OpenAI `text-embedding-3-small`). |
| 2. Vector database | `rag/vector_db.py` | Stores each chunk with its vector and finds the nearest vectors to a question (Chroma, saved in `chroma_db/`). |
| 3. Retrieval | `rag/retriever.py` | Embeds the question, then asks the vector database for the 10 closest chunks. Fast, but rough. |
| 4. Re-ranking | `rag/reranker.py` | A cross-encoder reads the question and each chunk *together* and scores relevance from 0 to 1. Slow but precise, so it only re-scores the top 10 and keeps 3 (`BAAI/bge-reranker-base`, runs locally). |
| 5. LLM engine | `rag/llm.py` | Writes the answer from the 3 chunks, and is told to say "I don't know" if they don't contain it (OpenAI `gpt-4o-mini`). |

Supporting files:
- `rag/loader.py` reads `.txt`, `.md` and `.pdf` files and cuts them into overlapping chunks.
- `rag/pipeline.py` connects the steps.
- `config.py` holds every setting (models, chunk size, top-k, top-n).

## Run it

```bash
cd cortex-beginner
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env              # then put your OpenAI key in .env

.venv/bin/python ingest.py        # build the vector database from data/
.venv/bin/python ask.py "How long is the Kestrel-7 warranty?"
.venv/bin/python ask.py           # interactive mode
```

The first run of `ask.py` downloads the re-ranker model (about 1 GB) once. Running the sample
data costs a fraction of a cent in OpenAI usage.

## The sample data

`data/` describes **Veldrin Corp and its Kestrel-7 sensor, a company that doesn't exist**.
That's deliberate: if the system answers correctly, the answer can only have come from
retrieval, not from the LLM's memory. Try questions the documents can't answer, such as
"What is the CEO's home address?", and the system should reply "I don't know."

To use your own documents, drop `.txt`, `.md` or `.pdf` files into `data/` and run
`ingest.py` again.

## Tests

```bash
.venv/bin/python -m pytest
```

The tests need no API key and no model download. They use fakes for the embedder, the
re-ranker and the LLM, and the real chunker and vector database.

## What's deliberately left out

These are covered in cortex-core and cortex-langgraph:
- a relevance threshold that refuses when no chunk is good enough;
- smarter chunking on sentence and paragraph boundaries;
- PDF table extraction and OCR;
- chat memory and an agent with tools;
- a web API and UI;
- monitoring;
- evaluation of retrieval quality.
