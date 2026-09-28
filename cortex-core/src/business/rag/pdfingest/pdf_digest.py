"""Reusable PDF ingestion pipeline for RAG, powered by Docling.

Docling (https://github.com/docling-project/docling) parses layout, OCR, and
table structure fully locally — no external API, no API key, and no per-page
cost, unlike the Unstructured-API-based table extraction this module replaces.

Converts PDFs in a directory into structured artifacts:
- extracted text blocks (section headers, narrative text, list items)
- tables, extracted directly as Markdown (no image round-trip needed)
- combined text trimmed to a configurable character budget

Downstream steps (chunk → embed → index) can consume the returned
``IngestedDocument`` objects without dealing with side effects or prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


# Docling's DocItemLabel values (as plain strings) that correspond to the
# "Title / NarrativeText / ListItem" categories the rest of the pipeline
# expects. Page headers/footers, footnotes, and captions are dropped as noise.
USEFUL_LABELS = {"section_header", "title", "text", "list_item"}


@dataclass
class IngestedDocument:
    source_id: str
    text_blocks: List[str]
    table_text: str
    combined_text: str
    metadata: Dict[str, int]


def _build_converter(do_ocr: bool, do_table_structure: bool) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(do_ocr=do_ocr, do_table_structure=do_table_structure)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _load_docling_document(pdf_path: Path, strategy: str = "hi_res", extract_tables: bool = True):
    """Convert a PDF with Docling. strategy: 'hi_res' (OCR on, slower, better for scans) | 'fast' (no OCR)."""
    converter = _build_converter(do_ocr=(strategy == "hi_res"), do_table_structure=extract_tables)
    return converter.convert(str(pdf_path)).document


def _text_blocks_from_document(doc) -> List[str]:
    return [
        item.text.strip()
        for item in doc.texts
        if str(item.label) in USEFUL_LABELS and getattr(item, "text", None) and item.text.strip()
    ]


def _table_text_from_document(doc) -> str:
    table_texts = [table.export_to_markdown(doc) for table in doc.tables]
    return "\n\n".join(t for t in table_texts if t.strip())


def ingest_single_pdf(
    pdf_path: Path,
    image_output_dir: Optional[Path] = None,
    max_context_chars: int = 12_000,
    include_table_images: bool = True,
    pdf_strategy: str = "hi_res",
) -> IngestedDocument:
    """Ingest one PDF and return structured text artifacts.

    pdf_strategy: 'hi_res' (OCR on, slower, needed for scanned pages) | 'fast' (no OCR).
    include_table_images: whether to run Docling's table structure recognition.
    image_output_dir: unused, kept for backward compatibility with older callers.
    """
    doc = _load_docling_document(pdf_path, strategy=pdf_strategy, extract_tables=include_table_images)

    text_blocks = _text_blocks_from_document(doc)
    table_text = _table_text_from_document(doc) if include_table_images else ""

    pdf_text_only = "\n\n".join(text_blocks)

    combined_text = pdf_text_only
    separator = (
        "\n\n" + "=" * 80 + "\n" + "TABLES FROM DOCUMENT IMAGES:\n" + "=" * 80 + "\n"
    )

    if table_text:
        combined_text = pdf_text_only + separator + table_text

    # Trim to budget. Table text is always preserved in full; body text is trimmed
    # to fill whatever budget remains. This prevents tables from being silently
    # dropped when the combined document exceeds max_context_chars.
    if len(combined_text) > max_context_chars:
        if table_text:
            table_block = separator + table_text
            available_for_pdf = max_context_chars - len(table_block)
            if available_for_pdf > 0:
                combined_text = pdf_text_only[:available_for_pdf] + table_block
            else:
                # table text alone already exceeds budget — keep it, trim body
                combined_text = table_text[:max_context_chars]
        else:
            combined_text = pdf_text_only[:max_context_chars]

    metadata = {
        "text_block_count": len(text_blocks),
        "table_count": len(doc.tables),
        "total_chars": len(combined_text),
    }

    return IngestedDocument(
        source_id=pdf_path.name,
        text_blocks=text_blocks,
        table_text=table_text,
        combined_text=combined_text,
        metadata=metadata,
    )


def ingest_directory(
    data_dir: Path,
    image_output_dir: Optional[Path] = None,
    max_context_chars: int = 12_000,
    include_table_images: bool = True,
    pdf_strategy: str = "hi_res",
) -> List[IngestedDocument]:
    """Ingest all PDFs in a directory and its subdirectories, and return
    structured results. pdf_strategy: 'hi_res' (OCR on, slower) | 'fast' (no OCR).
    """
    pdf_files = list(Path(data_dir).rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir} (searched recursively)")

    results: List[IngestedDocument] = []
    for pdf_path in pdf_files:
        results.append(
            ingest_single_pdf(
                pdf_path=pdf_path,
                max_context_chars=max_context_chars,
                include_table_images=include_table_images,
                pdf_strategy=pdf_strategy,
            )
        )
    return results
