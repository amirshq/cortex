"""PDF ingestion for RAG — LangChain `DoclingLoader` edition.

Docling (https://github.com/docling-project/docling) still does the actual
work — layout analysis, OCR (RapidOCR), and table structure — fully locally,
no API key. What changed is the wrapper: instead of walking Docling's object
model by hand, `langchain_docling.DoclingLoader` turns each PDF into a
LangChain `Document`, the unit every other LangChain component (splitters,
vector stores, retrievers) consumes.

Differences from cortex-core, all deliberate:
- Tables are exported as Markdown *inline, where they appear in the document*,
  instead of being appended after a "TABLES FROM DOCUMENT IMAGES" separator.
  Each table now sits next to the text that explains it.
- Cleaning is done by Docling's own export filter (`labels=`): only titles,
  section headers, body text, list items and tables are kept — page headers,
  footers, footnotes and captions are dropped as noise, same as before.
- Document-level metadata is added by a custom `MetaExtractor`
  (source_id, page/table counts) — LangChain's hook for metadata enrichment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, DoclingDocument
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import BaseMetaExtractor, ExportType, MetaExtractor


# Docling item labels kept in the exported text. Page headers/footers,
# footnotes, and captions are dropped as noise (cleaning step).
USEFUL_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.TEXT,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.LIST_ITEM,
    # Kept on evidence from the golden-set evaluation: in technical books,
    # footnotes carry real facts ("TF32 actually has 19 bits") and captions
    # name the tables they label ("Table 2-4. The dimension values of …").
    # Dropping them made 25% of answerable questions unanswerable.
    DocItemLabel.FOOTNOTE,
    DocItemLabel.CAPTION,
}


class CortexMetaExtractor(MetaExtractor):
    """Document-level metadata enrichment.

    `source_id` (the file name) is the key the rest of the pipeline uses for
    provenance and stable chunk ids — the same key cortex-core used.
    """

    def extract_dl_doc_meta(self, file_path: str, dl_doc: DoclingDocument) -> Dict[str, Any]:
        return {
            "source": str(file_path),
            "source_id": Path(file_path).name,
            "page_count": len(dl_doc.pages),
            "table_count": len(dl_doc.tables),
        }


def needs_ocr(pdf_path: Path, sample_pages: int = 10, min_chars_per_page: int = 200) -> bool:
    """Does this PDF need OCR? Only if it has no real text layer (a scan).

    Born-digital PDFs (exported from Word, LaTeX, a publisher's tools) already
    contain their text; running OCR on them re-reads what is already there.
    Measured on a 535-page book: skipping OCR made conversion ~40% faster and
    lost only text inside figures (~4% of characters). We sample evenly spaced
    pages and call it a scan when they average under `min_chars_per_page`.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        count = len(pdf)
        if count == 0:
            return False
        picks = sorted({int(i * (count - 1) / max(sample_pages - 1, 1)) for i in range(min(sample_pages, count))})
        chars = [len(pdf[i].get_textpage().get_text_range().strip()) for i in picks]
        return sum(chars) / len(chars) < min_chars_per_page
    finally:
        pdf.close()


def _build_converter(do_ocr: bool, do_table_structure: bool) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(do_ocr=do_ocr, do_table_structure=do_table_structure)
    # FAST table model: same tables found on our benchmark, noticeably quicker
    # than ACCURATE on long documents.
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def build_pdf_loader(
    pdf_paths: List[Path],
    include_table_images: bool = True,
    pdf_strategy: str = "auto",
    meta_extractor: BaseMetaExtractor | None = None,
) -> DoclingLoader:
    """Configure a DoclingLoader for the given PDFs.

    pdf_strategy: 'auto' (OCR only for scanned PDFs — the default) |
                  'hi_res' (always OCR) | 'fast' (never OCR).
    include_table_images: whether to run Docling's table structure recognition
                          and keep tables in the exported text.
    """
    labels = set(USEFUL_LABELS)
    if include_table_images:
        labels.add(DocItemLabel.TABLE)

    return DoclingLoader(
        file_path=[str(p) for p in pdf_paths],
        converter=_build_converter(
            do_ocr=_resolve_ocr(pdf_strategy, pdf_paths), do_table_structure=include_table_images
        ),
        # MARKDOWN = one Document per PDF. Chunking is a separate, LangChain
        # step (see chunk.py) so its size/overlap stay under our control.
        export_type=ExportType.MARKDOWN,
        md_export_kwargs={"labels": labels, "image_placeholder": ""},
        meta_extractor=meta_extractor or CortexMetaExtractor(),
    )


def _resolve_ocr(pdf_strategy: str, pdf_paths: List[Path]) -> bool:
    if pdf_strategy == "hi_res":
        return True
    if pdf_strategy == "fast":
        return False
    if pdf_strategy == "auto":
        return any(needs_ocr(p) for p in pdf_paths)
    raise ValueError(f"Unknown pdf_strategy={pdf_strategy!r}. Use 'auto', 'hi_res' or 'fast'.")


def load_pdf_documents(
    data_dir: Path,
    max_context_chars: int | None = 12_000,
    include_table_images: bool = True,
    pdf_strategy: str = "auto",
) -> List[Document]:
    """Load every PDF under data_dir (recursively) as one LangChain Document each.

    Each document's text is trimmed to max_context_chars (None = keep it all).
    """
    pdf_files = sorted(Path(data_dir).rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir} (searched recursively)")

    loader = build_pdf_loader(
        pdf_files,
        include_table_images=include_table_images,
        pdf_strategy=pdf_strategy,
    )
    documents: List[Document] = []
    for doc in loader.lazy_load():
        text = doc.page_content.strip()
        if max_context_chars is not None:
            text = text[:max_context_chars]
        doc.page_content = text
        doc.metadata["total_chars"] = len(text)
        documents.append(doc)
    return documents
