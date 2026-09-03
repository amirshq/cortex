# PDF Ingestion Pipeline

PDF parsing runs on [Docling](https://github.com/docling-project/docling), IBM's open-source
document conversion library. Docling handles layout detection, OCR, and table structure
recognition entirely locally — no external API, no API key, and no per-page cost. This
replaced an earlier two-stage pipeline (the `unstructured` library plus a paid Unstructured
API call for table OCR).

This module extracts both text content and table data from PDF documents for use in the
RAG (Retrieval-Augmented Generation) pipeline.

## Overview

Docling converts a PDF into a single structured `DoclingDocument` in one pass:

```
PDF Document
    ↓
DocumentConverter().convert()
    ↓
DoclingDocument
    ├── .texts   → section headers, narrative text, list items
    └── .tables  → table structure, exported directly to Markdown
    ↓
Combined, budget-trimmed text
    ↓
Chunk → embed → store in Chroma
```

## Components

### `pdf_digest.py`

The only module in this package. Responsibilities:

- Converts a PDF with `docling.document_converter.DocumentConverter`
- Filters `doc.texts` down to useful categories (`section_header`, `title`, `text`, `list_item`)
- Exports each `doc.tables` entry to Markdown via `table.export_to_markdown(doc)`
- Combines text + tables into one string, trimmed to `max_context_chars` (tables are
  always kept in full; body text is trimmed to fit whatever budget remains)

**Key options** (both map onto Docling's `PdfPipelineOptions`):
- `pdf_strategy`: `"hi_res"` (OCR on — needed for scanned pages, default) | `"fast"` (OCR off, faster, text-layer only)
- `include_table_images`: whether to run Docling's table structure recognition (`do_table_structure`)

**Usage:**
```python
from src.business.rag.pdfingest.pdf_digest import ingest_directory

docs = ingest_directory(data_dir=Path("src/business/rag/data"))
```

Or via the CLI wrapper:
```bash
python scripts/index_cli.py rebuild --data-dir src/business/rag/data
```

## Dependencies

- **docling** — PDF parsing, layout, OCR, and table extraction (all local)
  ```bash
  pip install docling
  ```
- **openai** — embeddings and chat generation
- **python-dotenv** — environment variable management

No `UNSTRUCTURED_API_KEY` or any other external API key is required for PDF ingestion.

## Limitations

1. **Context limits**: Document text is truncated to a configurable character budget (default 12,000 chars; the API upload path uses 500,000 to fit entire papers including tables).
2. **First-run latency**: Docling downloads its layout/table-structure models from the Hugging Face Hub on first use; subsequent runs use the local cache.
3. **Language support**: OCR quality depends on Docling's bundled OCR engine (RapidOCR by default); non-English documents may need a different OCR language configuration.
