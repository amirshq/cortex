"""Index builder — LangChain edition.

Load → split → store, each step a LangChain component:

    DoclingLoader               PDFs → Documents          (pdfingest/pdf_digest.py)
    RecursiveCharacterTextSplitter  Documents → chunks    (pdfingest/chunk.py)
    VectorStore.add_documents   chunks → embedded + stored (vector_store.py)

index_builder.build_index()                     ← this file: calls the three steps in order
    1. load_pdf_documents()    → pdfingest/pdf_digest.py   (DoclingLoader)
    2. split_documents()       → pdfingest/chunk.py        (RecursiveCharacterTextSplitter)
    3. create_vector_store()   → vector_store.py
       .add_documents(batch)                                (embed + upsert, in batches from config.yml)

build_index() orchestrates the entire indexing pipeline in three steps:

Load PDFs → load_pdf_documents() uses Docling to convert PDFs into LangChain Document objects.
Split documents → split_documents() divides them into chunks using RecursiveCharacterTextSplitter.
Embed + store → processes chunks in batches (rag.indexing.embedding_batch_size) and calls vstore.add_documents(), which embeds the chunks and upserts them into the vector store.


cortex-core embedded each batch itself and handed raw vectors to the store.
A LangChain vector store owns its embedding model, so add_documents() embeds
and upserts in one call.

The LangChain version is much smaller because LangChain abstracts and standardizes
many of the low-level document-processing steps that you implemented manually in
the original version. Instead of directly inspecting the DoclingDocument, 
iterating through text blocks and tables, filtering labels, constructing
the combined text, and building your own IngestedDocument and metadata, 
DoclingLoader handles much of that conversion and returns a standard LangChain
Document. That Document can then directly flow into LangChain components
such as text splitters → embeddings → vector stores → retrievers, 
so you write less glue code while still using Docling underneath 
for the actual PDF parsing, OCR, layout analysis, and table extraction.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from .pdfingest.pdf_digest import load_pdf_documents
from .pdfingest.chunk import split_documents
from .vector_store import create_vector_store
from src.utils.config import rag_settings


def build_index(
    data_dir: Path,
    persist_dir: Path,
    max_context_chars: int | None = 12_000,
    include_table_images: bool = True,
    pdf_strategy: str = "auto",
    chunk_size: int | None = None,
    overlap: int | None = None,
    top_k_store: int | None = None,
) -> Tuple[int, int]:
    """
    Build or rebuild the vector index from PDFs.
    chunk_size / overlap default to config.yml → rag.chunking; the embedding
    batch size comes from rag.indexing.embedding_batch_size.
    Returns (docs_indexed, chunks_indexed).
    """
    load_dotenv()

    docs = load_pdf_documents(
        data_dir=data_dir,
        max_context_chars=max_context_chars,
        include_table_images=include_table_images,
        pdf_strategy=pdf_strategy,
    )
    print(f"📄 Chunking {len(docs)} document(s)...", flush=True)

    chunks = split_documents(docs, chunk_size=chunk_size, overlap=overlap)
    if top_k_store:
        chunks = chunks[:top_k_store]
    print(f"✅ Chunking complete: {len(chunks)} chunk(s)", flush=True)

    # One embedding API call embeds a whole batch, and fewer, larger calls are
    # much faster (measured: 50 chunks took 1.7s, 200 took 1.1s). OpenAI accepts
    # up to 2,048 inputs per call. Set in config.yml → rag.indexing.
    batch_size = rag_settings("indexing")["embedding_batch_size"]
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    vstore = None
    for i in range(0, len(chunks), batch_size):
        batch_num = i // batch_size + 1
        if vstore is None:
            print("🔗 Connecting to vector store...", flush=True)
            vstore = create_vector_store(persist_dir=str(persist_dir))
        print(f"📤 Embedding + storing batch {batch_num}/{total_batches}...", flush=True)
        batch = chunks[i : i + batch_size]
        # Explicit ids = the stable chunk ids, so re-indexing upserts in place.
        vstore.add_documents(batch, ids=[c.id for c in batch])
        print(f"✅ Batch {batch_num}/{total_batches} stored", flush=True)

    write_manifest(persist_dir, docs, chunks)
    return len(docs), len(chunks)


# ---------------------------------------------------------------------------
# Manifest: what is in the index
# ---------------------------------------------------------------------------
# A small JSON file next to the index listing the documents it holds, so the
# UI can show "AI Engineering.pdf — 535 pages, 1,412 chunks" after a reload
# without scanning the vector store (which Azure Search can't cheaply do).
MANIFEST_NAME = "documents.json"


def write_manifest(persist_dir: Path, docs, chunks) -> None:
    chunk_counts = Counter(c.metadata.get("source_id") for c in chunks)
    documents = [
        {
            "source_id": d.metadata.get("source_id"),
            "pages": d.metadata.get("page_count"),
            "chunks": chunk_counts.get(d.metadata.get("source_id"), 0),
            "indexed_at": datetime.utcnow().isoformat(),
        }
        for d in docs
    ]
    Path(persist_dir, MANIFEST_NAME).write_text(json.dumps(documents, indent=2))


def read_manifest(persist_dir: Path) -> List[Dict]:
    """Documents currently in the index ([] if nothing has been indexed)."""
    path = Path(persist_dir, MANIFEST_NAME)
    return json.loads(path.read_text()) if path.exists() else []
