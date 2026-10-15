"""Chunking — LangChain `RecursiveCharacterTextSplitter` edition.

cortex-core used a fixed character window: every chunk was exactly
`chunk_size` characters, cut wherever the window ended — often mid-word or
mid-sentence. `RecursiveCharacterTextSplitter` has the same size/overlap knobs
but tries to cut on paragraph breaks first, then lines, then sentences, then
words, so chunks end on natural boundaries.

Kept from cortex-core:
- Same defaults (800 chars, 100 overlap) — now read from config.yml → rag.chunking.
- Stable, content-derived chunk ids (sha1 of source + offset + text prefix),
  so re-indexing the same PDF overwrites chunks instead of duplicating them.
- Chunk metadata: source_id, chunk_start/chunk_end offsets, chunk_strategy,
  and a `section` tag ("table" vs "text").

Two strategies (config.yml → rag.chunking.strategy):
- "recursive"    the splitter above over the whole text, tables included. A long
                 table is cut wherever the size limit falls, so a row can land
                 in a chunk without its header ("| Llama 3-405B | 126 |": 126 of what?)
                 and the caption naming the table ends up in another chunk.
- "table_aware"  text goes through the same splitter, but each Markdown table is
                 split by ROWS, and every piece repeats the table's caption and
                 header row. Docling's cell padding (runs of spaces) is removed.
                 Found by the golden-set evaluation (experiment 3.1b).
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import rag_settings


STRATEGIES = ("recursive", "table_aware")


def _resolve(chunk_size: int | None, overlap: int | None) -> tuple[int, int]:
    """Explicit arguments win; anything not given comes from config.yml → rag.chunking."""
    cfg = rag_settings("chunking")
    return (cfg["chunk_size"] if chunk_size is None else chunk_size,
            cfg["chunk_overlap"] if overlap is None else overlap)


def _resolve_strategy(strategy: str | None) -> str:
    strategy = strategy or rag_settings("chunking").get("strategy", "recursive")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy={strategy!r}. Use one of {STRATEGIES}.")
    return strategy


def build_text_splitter(chunk_size: int | None = None, overlap: int | None = None) -> RecursiveCharacterTextSplitter:
    chunk_size, overlap = _resolve(chunk_size, overlap)
    assert overlap < chunk_size, "overlap must be smaller than chunk_size"
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        # Records each chunk's character offset in the source text as
        # metadata["start_index"] — what cortex-core called chunk_start.
        add_start_index=True,
    )


def split_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
    strategy: str | None = None,
) -> List[Document]:
    """Split loaded PDF documents into chunks with stable ids and provenance metadata.

    chunk_size / overlap / strategy default to config.yml → rag.chunking.
    Each returned Document has `id` set, so vector stores upsert by it.
    """
    chunk_size, overlap = _resolve(chunk_size, overlap)
    strategy = _resolve_strategy(strategy)
    splitter = build_text_splitter(chunk_size=chunk_size, overlap=overlap)
    strategy_name = f"{strategy}_{chunk_size}_overlap_{overlap}"

    if strategy == "recursive":
        chunks = splitter.split_documents(documents)
    else:
        chunks = [c for doc in documents for c in _split_table_aware(doc, splitter, chunk_size)]
    for chunk in chunks:
        start = int(chunk.metadata.pop("start_index", 0))
        chunk.metadata.update({
            "chunk_start": start,
            "chunk_end": start + len(chunk.page_content),
            "chunk_strategy": strategy_name,
            "section": "table" if is_table_chunk(chunk.page_content) else "text",
        })
        # Vector stores only accept flat, scalar metadata values.
        chunk.metadata = {k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))}
        chunk.id = build_chunk_id(chunk.page_content, start, chunk.metadata)
    return chunks


# ---------------------------------------------------------------------------
# Table-aware splitting
# ---------------------------------------------------------------------------
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_ROW = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_CAPTION = re.compile(r"^\s*Table\s+[\w.-]+[.:]")


def _blocks(text: str) -> List[Tuple[str, int, str]]:
    """Cut text into ("text" | "table", offset, content) blocks. A table block is
    a run of consecutive Markdown table rows."""
    blocks, kind, start, lines = [], None, 0, []
    offset = 0
    for line in text.splitlines(keepends=True):
        this = "table" if _TABLE_ROW.match(line) else "text"
        if this != kind:
            if lines:
                blocks.append((kind, start, "".join(lines)))
            kind, start, lines = this, offset, []
        lines.append(line)
        offset += len(line)
    if lines:
        blocks.append((kind, start, "".join(lines)))
    return blocks


def _compact_row(row: str) -> str:
    """Docling pads cells to align columns: "|   1 | Clarify   ...   |" → "| 1 | Clarify ... |"."""
    if _SEPARATOR_ROW.match(row):
        return "|" + "|".join("---" for _ in row.strip().strip("|").split("|")) + "|"
    return re.sub(r" {2,}", " ", row.strip())


def _caption_before(text_block: str) -> str:
    """The table's caption, if the paragraph right before the table is one ("Table 2-4. ...")."""
    paragraphs = [p.strip() for p in text_block.strip().split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs and _CAPTION.match(paragraphs[-1]) else ""


def _split_table(rows: List[str], caption: str, chunk_size: int) -> List[Tuple[int, str]]:
    """Pack rows into chunks of at most chunk_size characters; every chunk starts
    with the caption and the header. Returns (index of first row, text) pairs."""
    has_header = len(rows) > 1 and _SEPARATOR_ROW.match(rows[1])
    header = rows[:2] if has_header else []
    body = rows[2:] if has_header else rows
    prefix = "\n".join(([caption] if caption else []) + header)
    pieces, current, first = [], [], len(header)
    for i, row in enumerate(body, start=len(header)):
        candidate = "\n".join(([prefix] if prefix else []) + current + [row])
        if current and len(candidate) > chunk_size:
            pieces.append((first, "\n".join(([prefix] if prefix else []) + current)))
            current, first = [], i
        current.append(row)
    if current or not pieces:
        pieces.append((first, "\n".join(([prefix] if prefix else []) + current)))
    return pieces


def _split_table_aware(doc: Document, splitter: RecursiveCharacterTextSplitter, chunk_size: int) -> List[Document]:
    chunks: List[Document] = []
    previous_text = ""
    for kind, offset, content in _blocks(doc.page_content):
        if kind == "text":
            previous_text = content
            if content.strip():
                for c in splitter.split_documents([Document(page_content=content, metadata=dict(doc.metadata))]):
                    c.metadata["start_index"] = offset + int(c.metadata.get("start_index", 0))
                    chunks.append(c)
            continue
        lines = content.splitlines(keepends=True)
        line_offsets = [offset + sum(len(x) for x in lines[:i]) for i in range(len(lines))]
        rows = [_compact_row(r) for r in lines]
        for first, text in _split_table(rows, _caption_before(previous_text), chunk_size):
            chunks.append(Document(page_content=text,
                                   metadata={**doc.metadata, "start_index": line_offsets[first]}))
    return chunks


def is_table_chunk(text: str) -> bool:
    """A chunk is tagged "table" when most of its lines are Markdown table rows.

    Docling exports tables inline as Markdown (`| a | b |`), so the rows are
    easy to recognise. cortex-core could tag by offset instead because it
    appended every table after a fixed separator.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    table_rows = sum(1 for line in lines if line.startswith("|") and line.endswith("|"))
    return table_rows / len(lines) >= 0.5


def build_chunk_id(chunk_text: str, start: int, metadata: Dict) -> str:
    """Stable chunk id: same PDF + same offset + same text → same id."""
    base = (
        str(metadata.get("source_id", ""))  # PDF file name
        + str(start)
        + chunk_text[:100]
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()
