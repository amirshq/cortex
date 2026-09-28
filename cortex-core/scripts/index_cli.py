#You don’t need the CLI entrypoint if you don’t plan to trigger indexing from the terminal.
# It’s just a convenience wrapper around build_index() for ad‑hoc/manual runs (or automation/cron).
# If you always trigger indexing from code, you can ignore or remove scripts/index_cli.py.
"""
This is the CLI entrypoint to build or rebuild the RAG index using Terminal commands.

Use this to index a whole personal library of PDFs at once — drop every PDF you want
searchable into one folder, then run:

python scripts/index_cli.py --data-dir /path/to/your/pdf/library

Chunk IDs are content-addressed (hash of source filename + position + text), so
re-running this after adding new files to the same folder is safe and incremental:
unchanged files re-upsert identical chunks (no duplicates), new files just get added.
Editing an existing PDF's content does NOT remove its old chunks, though — that's a
known gap for a rare case (add/remove files, not edit-in-place).

Every argument is optional — with none given, it indexes src/business/rag/data (the
bundled sample PDF) into the same location the API/UI query. --persist-dir defaults
to whatever the API itself uses (src.business.rag.rag_persist_dir()) specifically so
CLI-indexed content and /api/v1/rag/query never disagree about where the index lives.

Full example:

python scripts/index_cli.py \
  --data-dir src/business/rag/data \
  --max-context-chars 12000 \
  --chunk-size 800 \
  --overlap 100 \
  --include-table-images true

This ingests your PDFs, chunks them, embeds them, and saves the index to the specified persist_dir.

IMPORTANT: the /api/v1/rag/upload endpoint (and the UI's PDF-upload dropzone) deletes
ALL previously indexed PDFs before indexing the one you just uploaded — it's built for
"replace with this one document," not "add to my library." Using it after building a
library with this CLI will wipe that library. Add new PDFs to your library by dropping
them in the folder and re-running this CLI, not through the upload endpoint.
"""
import sys
from pathlib import Path
from typing import Optional

import typer

# Ensure project root on sys.path when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.business.rag import rag_persist_dir
from src.business.rag.index_builder import build_index


def rebuild(
    data_dir: Path = typer.Option("src/business/rag/data", help="Directory with PDFs (your whole library, not just one file)"),
    persist_dir: Optional[Path] = typer.Option(
        None,
        help="Vector-store persistence directory. Defaults to the same location the API uses — "
        "override only if you deliberately want a separate index.",
    ),
    max_context_chars: int = typer.Option(12_000, help="Max combined text per document"),
    include_table_images: bool = typer.Option(True, help="Extract table structure via Docling (runs locally, no API key needed)"),
    pdf_strategy: str = typer.Option("hi_res", help="PDF strategy: 'hi_res' (OCR, slower) | 'fast' (no OCR, faster)"),
    chunk_size: int = typer.Option(800, help="Chunk size (characters)"),
    overlap: int = typer.Option(100, help="Chunk overlap (characters)"),
):
    resolved_persist_dir = persist_dir or rag_persist_dir()
    docs, chunks = build_index(
        data_dir=data_dir,
        persist_dir=resolved_persist_dir,
        max_context_chars=max_context_chars,
        include_table_images=include_table_images,
        pdf_strategy=pdf_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    typer.echo(f"Indexed {docs} document(s), {chunks} chunk(s) → {resolved_persist_dir}")


if __name__ == "__main__":
    typer.run(rebuild)
