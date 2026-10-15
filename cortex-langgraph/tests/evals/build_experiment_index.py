"""Build a separate RAG index for one experiment, without touching the app's index.

The app's index (data/rag_vectorstore/) stays as it is. Each experiment gets
its own Chroma index in data/experiments/<label>/, which eval_golden.py can
point at with --persist-dir.

Docling is the slow step (~3 min for a 535-page book) and its output doesn't
depend on chunking, so the converted documents are cached in
data/experiments/docling_cache/ and reused by every chunking variant. The cache
is keyed on the PDF's size and modification time; --refresh rebuilds it.

Usage (from cortex-langgraph/):
    .venv/bin/python tests/evals/build_experiment_index.py --label chunk-400 --chunk-size 400 --overlap 50
    .venv/bin/python tests/evals/build_experiment_index.py --label chunk-800-tables --strategy table_aware
    .venv/bin/python tests/evals/build_experiment_index.py --label embed-large --strategy table_aware \
        --embedding-model text-embedding-3-large

The embedding model is recorded in the index's experiment.json, and eval_golden.py
reads it from there, so questions are always embedded with the index's own model.

Cost: embedding the whole book once (~250k tokens ≈ $0.005 with text-embedding-3-small).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # cortex-langgraph/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langchain_core.documents import Document  # noqa: E402

from src.business.core.embedding import create_embedder  # noqa: E402
from src.business.rag.index_builder import write_manifest  # noqa: E402
from src.business.rag.pdfingest.chunk import split_documents  # noqa: E402
from src.business.rag.pdfingest.pdf_digest import load_pdf_documents  # noqa: E402
from src.business.rag.vector_store import create_vector_store  # noqa: E402
from src.utils.config import model_settings, rag_settings  # noqa: E402

EXPERIMENTS = ROOT / "data" / "experiments"
CACHE = EXPERIMENTS / "docling_cache"


def load_documents(pdf_dir: Path, refresh: bool) -> list[Document]:
    """Docling-converted documents, from the cache when the PDFs haven't changed."""
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    key = "|".join(f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in pdfs)
    cache_file = CACHE / "documents.json"
    if cache_file.exists() and not refresh:
        cached = json.loads(cache_file.read_text())
        if cached["key"] == key:
            print(f"📄 Using cached Docling output for {len(pdfs)} PDF(s)", flush=True)
            return [Document(page_content=d["text"], metadata=d["metadata"]) for d in cached["docs"]]

    print(f"📄 Running Docling on {len(pdfs)} PDF(s) (slow, cached afterwards)...", flush=True)
    t0 = time.perf_counter()
    # Same settings as the app's upload path (ingest_pdfs): no truncation, auto OCR.
    docs = load_pdf_documents(data_dir=pdf_dir, max_context_chars=None)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"key": key, "docs": [
        {"text": d.page_content, "metadata": d.metadata} for d in docs]}))
    print(f"✅ Docling done in {time.perf_counter() - t0:.0f}s, cached", flush=True)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="experiment name; the index goes in data/experiments/<label>/")
    parser.add_argument("--chunk-size", type=int, help="default: config.yml → rag.chunking.chunk_size")
    parser.add_argument("--overlap", type=int, help="default: config.yml → rag.chunking.chunk_overlap")
    parser.add_argument("--strategy", choices=["recursive", "table_aware"],
                        help="default: config.yml → rag.chunking.strategy")
    parser.add_argument("--embedding-model", help="default: config.yml → models.rag_embedding.name")
    parser.add_argument("--pdf-dir", default=str(ROOT / "data" / "rag_uploads"))
    parser.add_argument("--refresh", action="store_true", help="re-run Docling even if cached")
    args = parser.parse_args()

    docs = load_documents(Path(args.pdf_dir), args.refresh)
    chunks = split_documents(docs, chunk_size=args.chunk_size, overlap=args.overlap, strategy=args.strategy)
    strategy = chunks[0].metadata["chunk_strategy"] if chunks else "?"
    print(f"✂️  {len(chunks)} chunks ({strategy})", flush=True)

    persist_dir = EXPERIMENTS / args.label
    if persist_dir.exists():
        shutil.rmtree(persist_dir)  # a stale index would mix two chunkings
    persist_dir.mkdir(parents=True)

    batch = rag_settings("indexing")["embedding_batch_size"]
    embedding_model = args.embedding_model or model_settings("rag_embedding")["name"]
    store = create_vector_store(persist_dir=str(persist_dir), embedding=create_embedder(model=embedding_model))
    t0 = time.perf_counter()
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        store.add_documents(part, ids=[c.id for c in part])
        print(f"📤 embedded {min(i + batch, len(chunks))}/{len(chunks)}", flush=True)
    write_manifest(persist_dir, docs, chunks)

    sizes = sorted(len(c.page_content) for c in chunks)
    info = {"label": args.label, "chunk_strategy": strategy, "chunks": len(chunks),
            "embedding_model": embedding_model,
            "chunk_chars": {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]},
            "table_chunks": sum(c.metadata.get("section") == "table" for c in chunks),
            "built": time.strftime("%Y-%m-%d %H:%M"), "embed_seconds": round(time.perf_counter() - t0, 1)}
    (persist_dir / "experiment.json").write_text(json.dumps(info, indent=2))
    print(f"✅ {persist_dir.relative_to(ROOT)}: {json.dumps(info)}")


if __name__ == "__main__":
    main()
