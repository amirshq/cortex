"""Tests for ingestion: the Docling loader config, chunking, and build_index.

Docling itself (layout/OCR models) is never run here — that needs model
downloads. These tests pin OUR configuration of it and everything downstream.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from docling_core.types.doc import DocItemLabel
from langchain_core.documents import Document
from langchain_docling.loader import ExportType

from src.business.rag.index_builder import build_index
from src.business.rag.pdfingest.chunk import (
    build_chunk_id,
    build_text_splitter,
    is_table_chunk,
    split_documents,
)
from docling.datamodel.base_models import InputFormat

from src.business.rag.pdfingest.pdf_digest import (
    USEFUL_LABELS,
    CortexMetaExtractor,
    build_pdf_loader,
    load_pdf_documents,
)

PARAGRAPH = "This is a sentence about the project. " * 3


def doc(text: str, source_id: str = "a.pdf") -> Document:
    return Document(page_content=text, metadata={"source": f"/x/{source_id}", "source_id": source_id})


# ---------------------------------------------------------------------------
# Loader configuration
# ---------------------------------------------------------------------------
class TestPdfLoaderConfiguration:
    @pytest.fixture(autouse=True)
    def no_file_inspection(self, request):
        """These tests pass placeholder paths; the auto-OCR check would open
        them. Tests that are ABOUT that check patch it themselves."""
        if "auto" in request.node.name:
            yield
            return
        with patch("src.business.rag.pdfingest.pdf_digest.needs_ocr", return_value=False):
            yield

    def test_exports_one_markdown_document_per_pdf(self):
        loader = build_pdf_loader([Path("a.pdf")])
        assert loader._export_type == ExportType.MARKDOWN

    def test_cleaning_keeps_only_useful_labels_plus_tables(self):
        """Page headers, footers, footnotes and captions are noise."""
        labels = build_pdf_loader([Path("a.pdf")])._md_export_kwargs["labels"]
        assert labels == USEFUL_LABELS | {DocItemLabel.TABLE}
        for noise in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER,
                      DocItemLabel.FOOTNOTE, DocItemLabel.CAPTION):
            assert noise not in labels

    def test_tables_are_dropped_when_table_extraction_is_off(self):
        labels = build_pdf_loader([Path("a.pdf")], include_table_images=False)._md_export_kwargs["labels"]
        assert DocItemLabel.TABLE not in labels

    def test_image_placeholders_are_removed(self):
        assert build_pdf_loader([Path("a.pdf")])._md_export_kwargs["image_placeholder"] == ""

    @pytest.mark.parametrize("strategy, ocr", [("hi_res", True), ("fast", False)])
    def test_strategy_controls_ocr(self, strategy, ocr):
        with patch("src.business.rag.pdfingest.pdf_digest._build_converter") as build_converter:
            build_pdf_loader([Path("a.pdf")], pdf_strategy=strategy)
        assert build_converter.call_args.kwargs["do_ocr"] is ocr

    @pytest.mark.parametrize("scanned, ocr", [(True, True), (False, False)])
    def test_auto_strategy_runs_ocr_only_for_scanned_pdfs(self, scanned, ocr):
        with patch("src.business.rag.pdfingest.pdf_digest.needs_ocr", return_value=scanned), \
             patch("src.business.rag.pdfingest.pdf_digest._build_converter") as build_converter:
            build_pdf_loader([Path("a.pdf")], pdf_strategy="auto")
        assert build_converter.call_args.kwargs["do_ocr"] is ocr

    def test_auto_is_the_default(self):
        with patch("src.business.rag.pdfingest.pdf_digest.needs_ocr", return_value=False) as check, \
             patch("src.business.rag.pdfingest.pdf_digest._build_converter"):
            build_pdf_loader([Path("a.pdf")])
        check.assert_called_once()

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="pdf_strategy"):
            build_pdf_loader([Path("a.pdf")], pdf_strategy="turbo")

    def test_tables_use_the_fast_model(self):
        from docling.datamodel.pipeline_options import TableFormerMode

        from src.business.rag.pdfingest.pdf_digest import _build_converter

        converter = _build_converter(do_ocr=False, do_table_structure=True)
        options = converter.format_to_options[InputFormat.PDF].pipeline_options
        assert options.table_structure_options.mode == TableFormerMode.FAST

    def test_table_flag_controls_table_structure(self):
        with patch("src.business.rag.pdfingest.pdf_digest._build_converter") as build_converter:
            build_pdf_loader([Path("a.pdf")], include_table_images=False)
        assert build_converter.call_args.kwargs["do_table_structure"] is False


class TestMetadataEnrichment:
    def test_document_metadata(self):
        dl_doc = SimpleNamespace(pages={1: None, 2: None, 3: None}, tables=[object(), object()])
        meta = CortexMetaExtractor().extract_dl_doc_meta("/uploads/report.pdf", dl_doc)
        assert meta == {"source": "/uploads/report.pdf", "source_id": "report.pdf",
                        "page_count": 3, "table_count": 2}


class TestLoadPdfDocuments:
    def test_no_pdfs_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pdf_documents(tmp_path)

    def test_finds_pdfs_recursively_and_trims_to_budget(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.pdf").write_bytes(b"%PDF")
        loader = MagicMock()
        loader.lazy_load.return_value = iter([doc("  " + "x" * 50 + "  ")])
        with patch("src.business.rag.pdfingest.pdf_digest.build_pdf_loader", return_value=loader) as build:
            docs = load_pdf_documents(tmp_path, max_context_chars=10)

        assert build.call_args.args[0] == [tmp_path / "sub" / "a.pdf"]
        assert docs[0].page_content == "x" * 10
        assert docs[0].metadata["total_chars"] == 10

    def test_no_budget_keeps_the_whole_document(self, tmp_path):
        (tmp_path / "book.pdf").write_bytes(b"%PDF")
        loader = MagicMock()
        loader.lazy_load.return_value = iter([doc("x" * 2_000_000)])
        with patch("src.business.rag.pdfingest.pdf_digest.build_pdf_loader", return_value=loader):
            docs = load_pdf_documents(tmp_path, max_context_chars=None)
        assert len(docs[0].page_content) == 2_000_000


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
class TestSplitterConfiguration:
    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(AssertionError):
            build_text_splitter(chunk_size=100, overlap=100)

    def test_defaults_match_cortex_core(self):
        splitter = build_text_splitter()
        assert splitter._chunk_size == 800
        assert splitter._chunk_overlap == 100


class TestSplitDocuments:
    def test_empty_text_returns_no_chunks(self):
        assert split_documents([doc("")]) == []

    def test_whitespace_only_returns_no_chunks(self):
        assert split_documents([doc("   \n\n  ")]) == []

    def test_short_text_becomes_one_chunk(self):
        chunks = split_documents([doc("short text")])
        assert [c.page_content for c in chunks] == ["short text"]

    def test_long_text_is_split_within_the_size_limit(self):
        chunks = split_documents([doc(PARAGRAPH * 40)], chunk_size=200, overlap=20)
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 200 for c in chunks)

    def test_splits_on_paragraph_boundaries_first(self):
        """The reason for the switch: cortex-core's fixed window cut mid-word."""
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 20 for i in range(6))
        chunks = split_documents([doc(text)], chunk_size=200, overlap=0)
        assert all(c.page_content.startswith("Paragraph") for c in chunks)

    def test_offsets_point_back_into_the_source_text(self):
        text = PARAGRAPH * 20
        for chunk in split_documents([doc(text)], chunk_size=200, overlap=20):
            start, end = chunk.metadata["chunk_start"], chunk.metadata["chunk_end"]
            assert text[start:end] == chunk.page_content

    def test_every_chunk_carries_source_metadata(self):
        chunks = split_documents([doc(PARAGRAPH * 20, "report.pdf")], chunk_size=200, overlap=20)
        assert all(c.metadata["source_id"] == "report.pdf" for c in chunks)

    def test_every_chunk_records_the_strategy(self):
        chunks = split_documents([doc(PARAGRAPH)], chunk_size=300, overlap=30)
        assert chunks[0].metadata["chunk_strategy"] == "recursive_300_overlap_30"

    def test_start_index_is_renamed_to_chunk_start(self):
        chunk = split_documents([doc(PARAGRAPH)])[0]
        assert "start_index" not in chunk.metadata
        assert chunk.metadata["chunk_start"] == 0

    def test_metadata_is_flat_scalars_only(self):
        """Vector stores reject nested metadata values."""
        source = Document(page_content="text", metadata={"source_id": "a.pdf", "nested": {"x": 1}})
        chunk = split_documents([source])[0]
        assert "nested" not in chunk.metadata

    def test_multiple_documents_keep_their_own_source(self):
        chunks = split_documents([doc("one", "a.pdf"), doc("two", "b.pdf")])
        assert [c.metadata["source_id"] for c in chunks] == ["a.pdf", "b.pdf"]


class TestChunkIds:
    def test_every_chunk_has_an_id(self):
        assert all(c.id for c in split_documents([doc(PARAGRAPH * 20)], chunk_size=200, overlap=20))

    def test_ids_are_deterministic(self):
        first = [c.id for c in split_documents([doc(PARAGRAPH * 20)], chunk_size=200, overlap=20)]
        second = [c.id for c in split_documents([doc(PARAGRAPH * 20)], chunk_size=200, overlap=20)]
        assert first == second

    def test_different_sources_yield_different_ids(self):
        assert split_documents([doc("same", "a.pdf")])[0].id != split_documents([doc("same", "b.pdf")])[0].id

    def test_ids_within_a_document_are_unique(self):
        ids = [c.id for c in split_documents([doc(PARAGRAPH * 40)], chunk_size=200, overlap=20)]
        assert len(ids) == len(set(ids))

    def test_id_is_a_sha1_hex_digest(self):
        chunk_id = build_chunk_id("text", 0, {"source_id": "a.pdf"})
        assert len(chunk_id) == 40 and int(chunk_id, 16) >= 0


class TestTableSectionTagging:
    TABLE = "| Year | Count |\n|---|---|\n| 2020 | 10 |\n| 2021 | 12 |"

    def test_markdown_table_is_tagged_table(self):
        assert is_table_chunk(self.TABLE)

    def test_prose_is_tagged_text(self):
        assert not is_table_chunk(PARAGRAPH)

    def test_prose_mentioning_a_pipe_is_still_text(self):
        assert not is_table_chunk("Use a | b to pipe output.\nIt is common.")

    def test_empty_is_text(self):
        assert not is_table_chunk("")

    def test_chunks_get_the_section_tag(self):
        text = PARAGRAPH + "\n\n" + self.TABLE
        chunks = split_documents([doc(text)], chunk_size=120, overlap=0)
        sections = {c.page_content.splitlines()[0][:6]: c.metadata["section"] for c in chunks}
        assert "table" in sections.values() and "text" in sections.values()


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------
class TestBuildIndex:
    @pytest.fixture
    def pipeline(self):
        """Patch the loader and the store; keep the real splitter."""
        store = MagicMock()
        with patch("src.business.rag.index_builder.load_pdf_documents") as load, \
             patch("src.business.rag.index_builder.create_vector_store", return_value=store) as create:
            yield SimpleNamespace(load=load, create=create, store=store)

    def test_returns_document_and_chunk_counts(self, pipeline, tmp_path):
        pipeline.load.return_value = [doc(PARAGRAPH * 20)]
        docs, chunks = build_index(tmp_path, tmp_path, chunk_size=200, overlap=20)
        assert docs == 1
        assert chunks == sum(len(c.args[0]) for c in pipeline.store.add_documents.call_args_list)

    def test_no_documents_indexes_nothing(self, pipeline, tmp_path):
        pipeline.load.return_value = []
        assert build_index(tmp_path, tmp_path) == (0, 0)
        pipeline.create.assert_not_called()

    def test_stores_in_batches_of_five_hundred(self, pipeline, tmp_path):
        pipeline.load.return_value = [doc(f"chunk {i}", f"{i}.pdf") for i in range(1200)]
        build_index(tmp_path, tmp_path)
        sizes = [len(c.args[0]) for c in pipeline.store.add_documents.call_args_list]
        assert sizes == [500, 500, 200]

    def test_connects_to_the_store_once(self, pipeline, tmp_path):
        pipeline.load.return_value = [doc(f"chunk {i}", f"{i}.pdf") for i in range(1200)]
        build_index(tmp_path, tmp_path)
        pipeline.create.assert_called_once_with(persist_dir=str(tmp_path))

    def test_upserts_by_the_stable_chunk_ids(self, pipeline, tmp_path):
        pipeline.load.return_value = [doc("one", "a.pdf"), doc("two", "b.pdf")]
        build_index(tmp_path, tmp_path)
        call = pipeline.store.add_documents.call_args
        assert call.kwargs["ids"] == [d.id for d in call.args[0]]

    def test_top_k_store_truncates(self, pipeline, tmp_path):
        pipeline.load.return_value = [doc(f"chunk {i}", f"{i}.pdf") for i in range(10)]
        assert build_index(tmp_path, tmp_path, top_k_store=3) == (10, 3)

    def test_loader_options_are_forwarded(self, pipeline, tmp_path):
        pipeline.load.return_value = []
        build_index(tmp_path, tmp_path, max_context_chars=99, include_table_images=False, pdf_strategy="fast")
        assert pipeline.load.call_args.kwargs == {
            "data_dir": tmp_path, "max_context_chars": 99,
            "include_table_images": False, "pdf_strategy": "fast",
        }

    def test_chunk_size_and_overlap_are_forwarded(self, pipeline, tmp_path):
        pipeline.load.return_value = [doc(PARAGRAPH * 20)]
        build_index(tmp_path, tmp_path, chunk_size=150, overlap=10)
        stored = pipeline.store.add_documents.call_args.args[0]
        assert stored[0].metadata["chunk_strategy"] == "recursive_150_overlap_10"


class TestNeedsOcr:
    """Uses real PDFs generated on the fly with pypdfium2 (no fixtures on disk)."""

    def _pdf(self, path, text_per_page, pages=3):
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw

        pdf = pdfium.PdfDocument.new()
        font = raw.FPDFText_LoadStandardFont(pdf.raw, b"Helvetica")
        for _ in range(pages):
            page = pdf.new_page(612, 792)
            if text_per_page:
                obj = raw.FPDFPageObj_NewTextObj(pdf.raw, b"Helvetica", 10.0)
                import ctypes
                encoded = (text_per_page + "\x00").encode("utf-16-le")
                raw.FPDFText_SetText(obj, ctypes.cast(ctypes.c_char_p(encoded), ctypes.POINTER(raw.FPDF_WCHAR)))
                raw.FPDFPage_InsertObject(page.raw, obj)
                raw.FPDFPage_GenerateContent(page.raw)
        pdf.save(str(path))
        return path

    def test_text_pdf_does_not_need_ocr(self, tmp_path):
        from src.business.rag.pdfingest.pdf_digest import needs_ocr

        assert needs_ocr(self._pdf(tmp_path / "text.pdf", "word " * 100)) is False

    def test_pdf_without_text_needs_ocr(self, tmp_path):
        from src.business.rag.pdfingest.pdf_digest import needs_ocr

        assert needs_ocr(self._pdf(tmp_path / "scan.pdf", "")) is True
