"""Reusable PDF ingestion pipeline for RAG.

Converts PDFs in a directory into structured artifacts:
- extracted text blocks (title, narrative, list items, tables)
- optional table text from extracted images (via Unstructured API)
- combined text trimmed to a configurable character budget

Downstream steps (chunk → embed → index) can consume the returned
``IngestedDocument`` objects without dealing with side effects or prints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf


# Optional table-image processing (external API)
try:
    from .process_table_images import extract_table_text_from_images
    _TABLE_PROCESSING_AVAILABLE = True
except Exception:
    _TABLE_PROCESSING_AVAILABLE = False


load_dotenv()


USEFUL_CATEGORIES = {
    "Title",
    "NarrativeText",
    "ListItem",
    "Table",  # tables directly parsed from the PDF pages
}


@dataclass
class IngestedDocument:
    source_id: str
    text_blocks: List[str]
    table_text: str
    combined_text: str
    metadata: Dict[str, int]


def _load_pdf_elements(pdf_path: Path, image_output_dir: Path, strategy: str = "hi_res"):
    """Load PDF elements. strategy: 'hi_res' (OCR, slower, better quality) | 'fast' (no OCR, faster)."""
    return partition_pdf(
        filename=str(pdf_path),
        strategy=strategy,
        languages=["eng"],
        extract_images_in_pdf=(strategy == "hi_res"),  # fast mode often skips image extraction
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=str(image_output_dir),
    )


def _text_blocks_from_elements(elements) -> List[str]:
    return [
        el.text.strip()
        for el in elements
        if getattr(el, "category", None) in USEFUL_CATEGORIES and getattr(el, "text", None)
    ]


def _count_tables(elements) -> int:
    return len([
        el for el in elements if getattr(el, "category", None) == "Table" and getattr(el, "text", None)
    ])


def ingest_single_pdf(
    pdf_path: Path,
    image_output_dir: Path,
    max_context_chars: int = 12_000,
    include_table_images: bool = True,
    pdf_strategy: str = "hi_res",
) -> IngestedDocument:
    """Ingest one PDF and return structured text artifacts.
    pdf_strategy: 'hi_res' (OCR, slower) | 'fast' (no OCR, faster; table images may be skipped).
    """
    elements = _load_pdf_elements(pdf_path, image_output_dir=image_output_dir, strategy=pdf_strategy)

    text_blocks = _text_blocks_from_elements(elements)
    pdf_tables_count = _count_tables(elements)

    table_text = ""
    if include_table_images and _TABLE_PROCESSING_AVAILABLE:
        table_text = extract_table_text_from_images(image_output_dir)

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
        "table_count": pdf_tables_count,
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
    """Ingest all PDFs in a directory and return structured results.
    pdf_strategy: 'hi_res' (OCR, slower) | 'fast' (no OCR, faster).
    """
    if image_output_dir is None:
        image_output_dir = data_dir / ".." / "artifacts" / "images"
    image_output_dir = image_output_dir.resolve()
    image_output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    results: List[IngestedDocument] = []
    for pdf_path in pdf_files:
        results.append(
            ingest_single_pdf(
                pdf_path=pdf_path,
                image_output_dir=image_output_dir,
                max_context_chars=max_context_chars,
                include_table_images=include_table_images,
                pdf_strategy=pdf_strategy,
            )
        )
    return results

