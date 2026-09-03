"""Tests for API controllers."""

import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from fastapi import HTTPException, status

from src.api.controller import ChatController, RAGController, chat_controller, rag_controller
from src.database.dto import (
    ChatMessageRequest,
    ChatHistoryRequest,
    RAGQueryRequest,
    RAGUploadResponse,
)


class TestChatControllerSendMessage:
    """Test ChatController.send_message endpoint."""

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Successfully process a chat message."""
        request = ChatMessageRequest(
            message="Hello",
            session_id="session_123"
        )

        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "reply": "Hello! How can I help?",
                "model_used": "gpt-4o",
                "tokens_used": 42
            }

            with patch("src.api.controller.CHAT_MODEL_REQUESTS_TOTAL"):
                with patch("src.api.controller.CHAT_TOKENS_TOTAL"):
                    response = await ChatController.send_message(request)

            assert response.reply == "Hello! How can I help?"
            assert response.session_id == "session_123"
            assert response.model_used == "gpt-4o"
            assert response.tokens_used == 42

    @pytest.mark.asyncio
    async def test_send_message_empty_message(self):
        """Reject empty message."""
        request = ChatMessageRequest(
            message="",
            session_id="session_123"
        )

        with pytest.raises(HTTPException) as exc_info:
            await ChatController.send_message(request)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "cannot be empty" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_send_message_whitespace_only(self):
        """Reject whitespace-only message."""
        request = ChatMessageRequest(
            message="   \n  ",
            session_id="session_123"
        )

        with pytest.raises(HTTPException) as exc_info:
            await ChatController.send_message(request)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_send_message_records_metrics(self):
        """send_message records model and token metrics."""
        request = ChatMessageRequest(
            message="Test",
            session_id="session_123"
        )

        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "reply": "Response",
                "model_used": "gpt-4o",
                "tokens_used": 100
            }

            with patch("src.api.controller.CHAT_MODEL_REQUESTS_TOTAL") as mock_requests:
                with patch("src.api.controller.CHAT_TOKENS_TOTAL") as mock_tokens:
                    await ChatController.send_message(request)

                    mock_requests.labels.assert_called_once_with(model="gpt-4o")
                    mock_tokens.labels.assert_called_once_with(model="gpt-4o")

    @pytest.mark.asyncio
    async def test_send_message_handles_business_logic_error(self):
        """Handle errors from business logic."""
        request = ChatMessageRequest(
            message="Test",
            session_id="session_123"
        )

        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock) as mock_process:
            mock_process.side_effect = Exception("Database connection error")

            with pytest.raises(HTTPException) as exc_info:
                await ChatController.send_message(request)

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_send_message_handles_value_error(self):
        """Handle ValueError from business logic."""
        request = ChatMessageRequest(
            message="Test",
            session_id="session_123"
        )

        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock) as mock_process:
            mock_process.side_effect = ValueError("Invalid input format")

            with pytest.raises(HTTPException) as exc_info:
                await ChatController.send_message(request)

            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_send_message_without_tokens(self):
        """Handle response without tokens_used."""
        request = ChatMessageRequest(
            message="Test",
            session_id="session_123"
        )

        with patch("src.api.controller.process_chat_message", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "reply": "Response",
                "model_used": "gpt-4o",
            }

            with patch("src.api.controller.CHAT_MODEL_REQUESTS_TOTAL"):
                with patch("src.api.controller.CHAT_TOKENS_TOTAL") as mock_tokens:
                    response = await ChatController.send_message(request)

                    # Should not call tokens increment if tokens_used is None
                    mock_tokens.labels.return_value.inc.assert_not_called()


class TestChatControllerGetHistory:
    """Test ChatController.get_chat_history endpoint."""

    @pytest.mark.asyncio
    async def test_get_history_success(self):
        """Successfully retrieve chat history."""
        request = ChatHistoryRequest(
            user_id=1,
            session_id="session_123"
        )

        with patch("src.api.controller.get_chat_history", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"}
                ],
                "total": 2,
                "session_id": "session_123"
            }

            response = await ChatController.get_chat_history(request)

            assert len(response.messages) == 2
            assert response.total == 2
            assert response.session_id == "session_123"

    @pytest.mark.asyncio
    async def test_get_history_invalid_user_id(self):
        """Reject invalid user_id."""
        request = ChatHistoryRequest(
            user_id=0,
            session_id="session_123"
        )

        with pytest.raises(HTTPException) as exc_info:
            await ChatController.get_chat_history(request)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_get_history_negative_user_id(self):
        """Reject negative user_id."""
        request = ChatHistoryRequest(
            user_id=-1,
            session_id="session_123"
        )

        with pytest.raises(HTTPException) as exc_info:
            await ChatController.get_chat_history(request)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_get_history_empty_messages(self):
        """Handle empty message list."""
        request = ChatHistoryRequest(
            user_id=1,
            session_id="session_123"
        )

        with patch("src.api.controller.get_chat_history", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                "messages": [],
                "total": 0,
                "session_id": "session_123"
            }

            response = await ChatController.get_chat_history(request)

            assert response.messages == []
            assert response.total == 0

    @pytest.mark.asyncio
    async def test_get_history_reraises_http_exception(self):
        """Re-raise HTTPExceptions from business logic."""
        request = ChatHistoryRequest(
            user_id=1,
            session_id="session_123"
        )

        original_exc = HTTPException(status_code=403, detail="Forbidden")

        with patch("src.api.controller.get_chat_history", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = original_exc

            with pytest.raises(HTTPException) as exc_info:
                await ChatController.get_chat_history(request)

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_history_handles_other_errors(self):
        """Convert unexpected errors to HTTP 500."""
        request = ChatHistoryRequest(
            user_id=1,
            session_id="session_123"
        )

        with patch("src.api.controller.get_chat_history", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Database unavailable")

            with pytest.raises(HTTPException) as exc_info:
                await ChatController.get_chat_history(request)

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestRAGControllerQuery:
    """Test RAGController.query endpoint."""

    @pytest.mark.asyncio
    async def test_query_success(self):
        """Successfully execute RAG query."""
        request = RAGQueryRequest(question="What is AI?")

        with patch("src.api.controller.query_rag", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = {
                "answer": "AI is...",
                "sources": ["document1.pdf"]
            }

            response = await RAGController.query(request)

            assert response.answer == "AI is..."
            assert response.sources == ["document1.pdf"]

    @pytest.mark.asyncio
    async def test_query_empty_question(self):
        """Reject empty question."""
        request = RAGQueryRequest(question="")

        with pytest.raises(HTTPException) as exc_info:
            await RAGController.query(request)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "cannot be empty" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_query_whitespace_only(self):
        """Reject whitespace-only question."""
        request = RAGQueryRequest(question="   \n  ")

        with pytest.raises(HTTPException) as exc_info:
            await RAGController.query(request)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_query_handles_errors(self):
        """Convert errors from RAG pipeline to HTTP 500."""
        request = RAGQueryRequest(question="Test?")

        with patch("src.api.controller.query_rag", new_callable=AsyncMock) as mock_query:
            mock_query.side_effect = Exception("Vector store unavailable")

            with pytest.raises(HTTPException) as exc_info:
                await RAGController.query(request)

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "RAG query failed" in exc_info.value.detail


class TestRAGControllerUpload:
    """Test RAGController.upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_pdf_success(self):
        """Successfully upload and index PDF."""
        mock_file = MagicMock()
        mock_file.filename = "test.pdf"
        mock_file.file = MagicMock()

        with patch("src.api.controller.ingest_pdfs", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = {
                "docs_indexed": 1,
                "chunks_indexed": 25,
                "table_ocr_enabled": False
            }

            with patch("src.api.controller.RAG_DOCUMENTS_INDEXED_TOTAL"):
                with patch("src.api.controller.RAG_CHUNKS_INDEXED_TOTAL"):
                    with patch("pathlib.Path.mkdir"):
                        with patch("pathlib.Path.glob", return_value=[]):
                            with patch("shutil.copyfileobj"):
                                response = await RAGController.upload(mock_file)

            assert response.filename == "test.pdf"
            assert response.docs_indexed == 1
            assert response.chunks_indexed == 25

    @pytest.mark.asyncio
    async def test_upload_non_pdf_file(self):
        """Reject non-PDF files."""
        mock_file = MagicMock()
        mock_file.filename = "test.txt"

        with pytest.raises(HTTPException) as exc_info:
            await RAGController.upload(mock_file)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "PDF" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_upload_no_filename(self):
        """Reject files without filename."""
        mock_file = MagicMock()
        mock_file.filename = None

        with pytest.raises(HTTPException) as exc_info:
            await RAGController.upload(mock_file)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_upload_case_insensitive_extension(self):
        """Accept PDF with different case."""
        mock_file = MagicMock()
        mock_file.filename = "test.PDF"
        mock_file.file = MagicMock()

        with patch("src.api.controller.ingest_pdfs", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = {
                "docs_indexed": 1,
                "chunks_indexed": 10
            }

            with patch("src.api.controller.RAG_DOCUMENTS_INDEXED_TOTAL"):
                with patch("src.api.controller.RAG_CHUNKS_INDEXED_TOTAL"):
                    with patch("pathlib.Path.mkdir"):
                        with patch("pathlib.Path.glob", return_value=[]):
                            with patch("shutil.copyfileobj"):
                                response = await RAGController.upload(mock_file)

            assert response.filename == "test.PDF"

    @pytest.mark.asyncio
    async def test_upload_records_metrics(self):
        """Upload records metrics correctly."""
        mock_file = MagicMock()
        mock_file.filename = "test.pdf"
        mock_file.file = MagicMock()

        with patch("src.api.controller.ingest_pdfs", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = {
                "docs_indexed": 2,
                "chunks_indexed": 50,
                "table_ocr_enabled": True
            }

            with patch("src.api.controller.RAG_DOCUMENTS_INDEXED_TOTAL") as mock_docs:
                with patch("src.api.controller.RAG_CHUNKS_INDEXED_TOTAL") as mock_chunks:
                    with patch("pathlib.Path.mkdir"):
                        with patch("pathlib.Path.glob", return_value=[]):
                            with patch("shutil.copyfileobj"):
                                await RAGController.upload(mock_file)

            mock_docs.inc.assert_called_once_with(2)
            mock_chunks.inc.assert_called_once_with(50)

    @pytest.mark.asyncio
    async def test_upload_removes_old_pdfs(self):
        """Upload removes previously uploaded PDFs."""
        mock_file = MagicMock()
        mock_file.filename = "test.pdf"
        mock_file.file = MagicMock()

        mock_old_pdf = MagicMock()

        with patch("src.api.controller.ingest_pdfs", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.return_value = {
                "docs_indexed": 1,
                "chunks_indexed": 10
            }

            with patch("src.api.controller.RAG_DOCUMENTS_INDEXED_TOTAL"):
                with patch("src.api.controller.RAG_CHUNKS_INDEXED_TOTAL"):
                    with patch("pathlib.Path.mkdir"):
                        with patch("pathlib.Path.glob", return_value=[mock_old_pdf]):
                            with patch("shutil.copyfileobj"):
                                await RAGController.upload(mock_file)

            mock_old_pdf.unlink.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_handles_errors(self):
        """Convert upload errors to HTTP 500."""
        mock_file = MagicMock()
        mock_file.filename = "test.pdf"

        with patch("src.api.controller.ingest_pdfs", new_callable=AsyncMock) as mock_ingest:
            mock_ingest.side_effect = Exception("Indexing failed")

            with patch("src.api.controller.RAG_DOCUMENTS_INDEXED_TOTAL"):
                with patch("src.api.controller.RAG_CHUNKS_INDEXED_TOTAL"):
                    with patch("pathlib.Path.mkdir"):
                        with patch("pathlib.Path.glob", return_value=[]):
                            with patch("shutil.copyfileobj"):
                                with pytest.raises(HTTPException) as exc_info:
                                    await RAGController.upload(mock_file)

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "PDF upload failed" in exc_info.value.detail
