#You don’t need the CLI entrypoint if you don’t plan to trigger indexing from the terminal. 
# It’s just a convenience wrapper around build_index() for ad‑hoc/manual runs (or automation/cron). 
# If you always trigger indexing from code, you can ignore or remove scripts/index_cli.py.
"""
This is the CLI entrypoint to build or rebuild the RAG index using Terminal commands.
The rebuild command is the CLI entrypoint: you can run it from the terminal to 
rebuild the vector index with custom options. For example:

python scripts/rebuild_index.py rebuild \
  --data-dir src/business/rag/data \
  --persist-dir src/business/rag/vectorstore \
  --max-context-chars 12000 \
  --chunk-size 800 \
  --overlap 100 \
  --include-table-images true

This ingests your PDFs, chunks them, embeds them, and saves the index to the specified persist_dir.
"""
import sys
from pathlib import Path

import typer

# Ensure project root on sys.path when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.business.rag.index_builder import build_index

# persist_dir is where the Chroma vector DB is persisted.
# build_index() writes chunk IDs, embeddings, documents, and metadata there.
# Default: src/business/rag/vectorstore (configurable via CLI).

def rebuild(
    data_dir: Path = typer.Option("src/business/rag/data", help="Directory with PDFs"),
    persist_dir: Path = typer.Option("src/business/rag/vectorstore", help="Chroma persistence directory"),
    max_context_chars: int = typer.Option(12_000, help="Max combined text per document"),
    include_table_images: bool = typer.Option(True, help="Extract table structure via Docling (runs locally, no API key needed)"),
    pdf_strategy: str = typer.Option("hi_res", help="PDF strategy: 'hi_res' (OCR, slower) | 'fast' (no OCR, faster)"),
    chunk_size: int = typer.Option(800, help="Chunk size (characters)"),
    overlap: int = typer.Option(100, help="Chunk overlap (characters)"),
):
    docs, chunks = build_index(
        data_dir=data_dir,
        persist_dir=persist_dir,
        max_context_chars=max_context_chars,
        include_table_images=include_table_images,
        pdf_strategy=pdf_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    typer.echo(f"Indexed {docs} document(s), {chunks} chunk(s) → {persist_dir}")


if __name__ == "__main__":
    typer.run(rebuild)
