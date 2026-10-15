# PDF Ingestion Pipeline — LangChain edition

PDF parsing runs on [Docling](https://github.com/docling-project/docling), IBM's open-source
document conversion library. It does layout detection, OCR (RapidOCR) and table structure
recognition entirely locally, with no external API, no API key and no per-page cost.

In cortex-langgraph, Docling is driven through LangChain's
[`DoclingLoader`](https://github.com/docling-project/docling-langchain), which yields LangChain
`Document`s. A `Document` is the unit every other LangChain component consumes: splitters,
vector stores and retrievers.

## Overview

```
PDFs in data_dir (recursive)
    ↓
DoclingLoader (export_type=MARKDOWN)          pdf_digest.py
    ├── cleaning: keep title / section header / text / paragraph / list item / table;
    │             page headers, footers, footnotes, captions and image placeholders dropped
    ├── tables:   exported inline as Markdown, where they appear in the document
    └── metadata: CortexMetaExtractor → source, source_id, page_count, table_count
    ↓
one Document per PDF, trimmed to max_context_chars
    ↓
RecursiveCharacterTextSplitter (800 / 100)    chunk.py
    ├── cuts on paragraph → line → sentence → word boundaries
    ├── chunk_start / chunk_end offsets, chunk_strategy, section ("table" | "text")
    └── stable sha1 chunk id → Document.id
    ↓
VectorStore.add_documents(ids=...)            ../index_builder.py (embeds + upserts, batches of 50)
```

## Components

### `pdf_digest.py`

- `build_pdf_loader(paths, include_table_images, pdf_strategy)` configures the `DoclingLoader`.
- `load_pdf_documents(data_dir, max_context_chars, ...)` returns `List[Document]`, one per PDF.
- `CortexMetaExtractor` handles document-level metadata enrichment.

**Key options** (both map onto Docling's `PdfPipelineOptions`):
- `pdf_strategy`: `"hi_res"` (OCR on, needed for scanned pages; the default) or `"fast"`
  (OCR off, text layer only).
- `include_table_images`: whether to run table structure recognition and keep tables in the
  text.

### `chunk.py`

- `split_documents(documents, chunk_size=800, overlap=100)` returns chunk `Document`s with ids.
- `is_table_chunk(text)` returns true when most of a chunk's lines are Markdown table rows.
- `build_chunk_id(text, start, metadata)` gives the same PDF + offset + text the same id, so
  re-indexing upserts in place instead of duplicating chunks.

**Usage:**
```python
from src.business.rag.pdfingest.pdf_digest import load_pdf_documents
from src.business.rag.pdfingest.chunk import split_documents

chunks = split_documents(load_pdf_documents(Path("src/business/rag/data")))
```

Or via the CLI wrapper (from the project root, with this project's venv):
```bash
.venv/bin/python scripts/index_cli.py --data-dir src/business/rag/data
```

## Differences from cortex-core

- Tables are **inline**, next to the text that explains them. cortex-core appended all tables
  after a "TABLES FROM DOCUMENT IMAGES" separator and tagged `section` by character offset.
- Chunks end on natural boundaries instead of a fixed 800-character window.
- Budget trimming is a plain cut of the Markdown. cortex-core kept every table in full and
  trimmed only the body text. The API upload path uses a 500,000-character budget, so this
  only matters for the CLI's 12,000 default.

## Limitations

1. **Context limits**: document text is truncated to a character budget (12,000 chars by
   default; the API upload path uses 500,000).
2. **First-run latency**: Docling downloads its layout and table-structure models from the
   Hugging Face Hub on first use; later runs use the local cache.
3. **Language support**: OCR quality depends on Docling's bundled OCR engine (RapidOCR by
   default). Non-English documents may need a different OCR language configuration.
