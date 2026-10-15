"""Controller branches the original unit tests never executed.

Branch coverage on src/api/controller.py showed three untested paths. Each is
real behaviour, not padding:

- ChatController.send_message's cost branch. It runs only when the business
  layer reports input/output token counts. NOTE: process_chat_message never
  returns those today, so in production this branch is unreachable and
  chat_cost_total never moves. These tests pin the controller's side of the
  contract so costing works the day token counts are wired through.
- send_message's non-dict return. The business layer may hand back a finished
  ChatMessageResponse, which must reach the caller untouched.
- RAGController.upload's `except HTTPException: raise`. Without it, a 4xx
  raised during ingestion would be caught by the generic handler and
  re-reported to the client as a 500.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status

from src.api.controller import ChatController, RAGController
from src.database.dto import ChatMessageRequest, ChatMessageResponse


def business_result(**overrides):
    """A process_chat_message result that includes token counts."""
    result = {
        "reply": "Hello! How can I help?",
        "model_used": "gpt-4o",
        "tokens_used": 150,
        "input_tokens": 100,
        "output_tokens": 50,
    }
    result.update(overrides)
    return result


class TestSendMessageCostBranch:
    @pytest.mark.asyncio
    async def test_cost_is_calculated_from_the_reported_token_counts(self):
        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock,
                   return_value=business_result()), \
             patch("src.api.controller.calculate_chat_cost", return_value=0.0125) as calc, \
             patch("src.api.controller.CHAT_MODEL_REQUESTS_TOTAL"), \
             patch("src.api.controller.CHAT_TOKENS_TOTAL"), \
             patch("src.api.controller.CHAT_COST_TOTAL"):
            await ChatController.send_message(ChatMessageRequest(message="Hello"))

        calc.assert_called_once_with(
            model_name="gpt-4o", input_tokens=100, output_tokens=50, provider="openai")

    @pytest.mark.asyncio
    async def test_provider_comes_from_llm_provider_normalised(self, monkeypatch):
        """Pricing tables are keyed by provider; a padded or upper-case env
        value must still select the right table."""
        monkeypatch.setenv("LLM_PROVIDER", "  Azure_OpenAI ")
        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock,
                   return_value=business_result()), \
             patch("src.api.controller.calculate_chat_cost", return_value=0.01) as calc, \
             patch("src.api.controller.CHAT_MODEL_REQUESTS_TOTAL"), \
             patch("src.api.controller.CHAT_TOKENS_TOTAL"), \
             patch("src.api.controller.CHAT_COST_TOTAL"):
            await ChatController.send_message(ChatMessageRequest(message="Hello"))

        assert calc.call_args.kwargs["provider"] == "azure_openai"

    @pytest.mark.asyncio
    async def test_positive_cost_is_added_to_chat_cost_total_for_that_model(self):
        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock,
                   return_value=business_result(model_used="gpt-4o-mini")), \
             patch("src.api.controller.calculate_chat_cost", return_value=0.0125), \
             patch("src.api.controller.CHAT_MODEL_REQUESTS_TOTAL"), \
             patch("src.api.controller.CHAT_TOKENS_TOTAL"), \
             patch("src.api.controller.CHAT_COST_TOTAL") as cost_total:
            await ChatController.send_message(ChatMessageRequest(message="Hello"))

        cost_total.labels.assert_called_once_with(model="gpt-4o-mini")
        cost_total.labels.return_value.inc.assert_called_once_with(0.0125)

    @pytest.mark.asyncio
    async def test_zero_cost_is_not_recorded(self):
        """Unknown models price at $0 by design (see cost.py); recording a
        zero would create an empty per-model series for nothing."""
        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock,
                   return_value=business_result(model_used="unpriced-model")), \
             patch("src.api.controller.calculate_chat_cost", return_value=0.0), \
             patch("src.api.controller.CHAT_MODEL_REQUESTS_TOTAL"), \
             patch("src.api.controller.CHAT_TOKENS_TOTAL"), \
             patch("src.api.controller.CHAT_COST_TOTAL") as cost_total:
            await ChatController.send_message(ChatMessageRequest(message="Hello"))

        cost_total.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_output_tokens_alone_are_enough_to_trigger_costing(self):
        """The guard is `input_tokens or output_tokens`, not `and`."""
        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock,
                   return_value=business_result(input_tokens=0, output_tokens=40)), \
             patch("src.api.controller.calculate_chat_cost", return_value=0.002) as calc, \
             patch("src.api.controller.CHAT_MODEL_REQUESTS_TOTAL"), \
             patch("src.api.controller.CHAT_TOKENS_TOTAL"), \
             patch("src.api.controller.CHAT_COST_TOTAL"):
            await ChatController.send_message(ChatMessageRequest(message="Hello"))

        calc.assert_called_once()
        assert calc.call_args.kwargs["output_tokens"] == 40

    @pytest.mark.asyncio
    async def test_without_token_counts_costing_is_skipped_entirely(self):
        """This is the shape process_chat_message actually returns today."""
        today = {"reply": "Hi", "model_used": "gpt-4o", "tokens_used": None}
        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock,
                   return_value=today), \
             patch("src.api.controller.calculate_chat_cost") as calc, \
             patch("src.api.controller.CHAT_MODEL_REQUESTS_TOTAL"), \
             patch("src.api.controller.CHAT_TOKENS_TOTAL"), \
             patch("src.api.controller.CHAT_COST_TOTAL") as cost_total:
            await ChatController.send_message(ChatMessageRequest(message="Hello"))

        calc.assert_not_called()
        cost_total.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_costing_does_not_change_the_response(self):
        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock,
                   return_value=business_result()), \
             patch("src.api.controller.calculate_chat_cost", return_value=0.0125), \
             patch("src.api.controller.CHAT_MODEL_REQUESTS_TOTAL"), \
             patch("src.api.controller.CHAT_TOKENS_TOTAL"), \
             patch("src.api.controller.CHAT_COST_TOTAL"):
            response = await ChatController.send_message(
                ChatMessageRequest(message="Hello", session_id="s1"))

        assert isinstance(response, ChatMessageResponse)
        assert (response.reply, response.session_id, response.model_used, response.tokens_used) == \
               ("Hello! How can I help?", "s1", "gpt-4o", 150)


class TestSendMessageNonDictResult:
    @pytest.mark.asyncio
    async def test_a_prebuilt_response_is_returned_unchanged(self):
        prebuilt = ChatMessageResponse(reply="already built", session_id="s9",
                                       model_used="gpt-4o", tokens_used=7)
        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock,
                   return_value=prebuilt):
            response = await ChatController.send_message(
                ChatMessageRequest(message="Hello", session_id="s9"))

        assert response is prebuilt


class TestUploadHTTPExceptionPassthrough:
    @staticmethod
    def pdf_upload(name: str = "report.pdf"):
        upload = MagicMock()
        upload.filename = name
        upload.file = io.BytesIO(b"%PDF-1.4\n% minimal test body\n")
        return upload

    @pytest.mark.asyncio
    async def test_a_4xx_from_ingestion_keeps_its_status_and_detail(self):
        original = HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                 detail="PDF is encrypted")
        with patch("src.api.controller.ingest_pdfs", new_callable=AsyncMock, side_effect=original), \
             patch("src.api.controller.RAG_DOCUMENTS_INDEXED_TOTAL"), \
             patch("src.api.controller.RAG_CHUNKS_INDEXED_TOTAL"):
            with pytest.raises(HTTPException) as exc_info:
                await RAGController.upload(self.pdf_upload())

        assert exc_info.value is original
        assert exc_info.value.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        assert exc_info.value.detail == "PDF is encrypted"

    @pytest.mark.asyncio
    async def test_a_4xx_from_ingestion_is_not_rewrapped_as_a_500(self):
        with patch("src.api.controller.ingest_pdfs", new_callable=AsyncMock,
                   side_effect=HTTPException(status_code=409, detail="index busy")), \
             patch("src.api.controller.RAG_DOCUMENTS_INDEXED_TOTAL"), \
             patch("src.api.controller.RAG_CHUNKS_INDEXED_TOTAL"):
            with pytest.raises(HTTPException) as exc_info:
                await RAGController.upload(self.pdf_upload())

        assert exc_info.value.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "PDF upload failed" not in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_a_failed_ingestion_records_no_indexing_metrics(self):
        with patch("src.api.controller.ingest_pdfs", new_callable=AsyncMock,
                   side_effect=HTTPException(status_code=415, detail="PDF is encrypted")), \
             patch("src.api.controller.RAG_DOCUMENTS_INDEXED_TOTAL") as docs, \
             patch("src.api.controller.RAG_CHUNKS_INDEXED_TOTAL") as chunks:
            with pytest.raises(HTTPException):
                await RAGController.upload(self.pdf_upload())

        docs.inc.assert_not_called()
        chunks.inc.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_writes_only_inside_the_sandbox(self, sandboxed_paths):
        """Guards the fixture itself: if _PROJECT_ROOT stopped being redirected,
        this upload would land in the real data/rag_uploads and delete the
        PDFs already there."""
        with patch("src.api.controller.ingest_pdfs", new_callable=AsyncMock,
                   return_value={"docs_indexed": 1, "chunks_indexed": 3}):
            await RAGController.upload(self.pdf_upload("sandboxed.pdf"))

        assert (sandboxed_paths.uploads_dir / "sandboxed.pdf").exists()
