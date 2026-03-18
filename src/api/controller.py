import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from typing import Optional

from src.database.dto import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatHistoryRequest,
    ChatHistoryResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGUploadResponse,
)
from src.business.chatbot import process_chat_message, get_chat_history
from src.business.rag import query_rag, ingest_pdfs

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ChatController:
    @staticmethod
    async def send_message(request: ChatMessageRequest) -> ChatMessageResponse:
        try:
            # Step 1: Validate additional business rules (if needed)
            # DTO already validates structure, but you can add domain validation here
            if not request.message or not request.message.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Message cannot be empty"
                )
            
            # Step 2: Call business logic
            # This is where the actual chatbot processing happens
            result = await process_chat_message(request)
            
            # Step 3: Convert business logic result to response DTO
            # The business logic returns a dict, we convert it to ChatMessageResponse
            if isinstance(result, dict):
                # Extract reply from result
                reply = result.get("reply", "I'm sorry, I couldn't process that.")
                
                return ChatMessageResponse(
                    reply=reply,
                    session_id=request.session_id,
                    model_used=result.get("model_used", "zephyr-7b-beta"),
                    tokens_used=result.get("tokens_used")
                )
            else:
                # If business logic already returns ChatMessageResponse
                return result
                
        except ValueError as e:
            # Handle validation errors from business logic
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            # Handle unexpected errors
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}"
            )
    
    @staticmethod
    async def get_chat_history(request: ChatHistoryRequest) -> ChatHistoryResponse:
        """
        Handle chat history retrieval requests.
        
        Flow:
        1. Validate request (user_id, pagination params)
        2. Call business logic to fetch from database
        3. Format response as ChatHistoryResponse
        4. Return response
        
        Note: History should come from database via business logic,
        NOT hardcoded in the controller (follows clean architecture).
        """
        try:
            # Step 1: Validate request
            if request.user_id <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid user_id"
                )
            
            # Step 2: Fetch history from Redis via business logic
            result = await get_chat_history(request)

            # Step 3: Return response
            return ChatHistoryResponse(
                messages=result["messages"],
                total=result["total"],
                session_id=result["session_id"],
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions (already formatted)
            raise
        except Exception as e:
            # Handle unexpected errors
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve chat history: {str(e)}"
            )


# Create a singleton instance (optional, but convenient)
chat_controller = ChatController()


# ---------------------------------------------------------------------------
# RAG Controller
# ---------------------------------------------------------------------------

class RAGController:
    @staticmethod
    async def query(request: RAGQueryRequest) -> RAGQueryResponse:
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty",
            )
        try:
            result = await query_rag(request.question)
            return RAGQueryResponse(answer=result["answer"], sources=result["sources"])
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"RAG query failed: {str(e)}",
            )

    @staticmethod
    async def upload(file: UploadFile) -> RAGUploadResponse:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are accepted",
            )
        try:
            uploads_dir = _PROJECT_ROOT / "data" / "rag_uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)

            dest = uploads_dir / file.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(file.file, f)

            result = await ingest_pdfs(uploads_dir)
            return RAGUploadResponse(
                filename=file.filename,
                docs_indexed=result["docs_indexed"],
                chunks_indexed=result["chunks_indexed"],
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PDF upload failed: {str(e)}",
            )


rag_controller = RAGController()



"""
Controller Layer - Handles HTTP requests/responses and coordinates business logic.

In Clean Architecture:
- Router (router.py) → Defines HTTP endpoints
- Controller (controller.py) → Handles request/response, validation, error handling
- Business Logic (business/chatbot.py) → Contains actual processing logic

The controller acts as a bridge between HTTP layer and business layer.
Chatcontroller Responsibilities:
1. Receive HTTP requests (via DTOs)
2. Validate input (DTOs handle this, but can add extra validation)
3. Call business logic
4. Handle errors and convert to HTTP responses
5. Return response DTOs

sendmessage Responsibilities:
        1. Receive ChatMessageRequest (validated by FastAPI)
        2. Call business logic to process the message
        3. Handle any errors
        4. Return ChatMessageResponse
        
        Args:
            request: ChatMessageRequest containing user message and metadata
            
        Returns:
            ChatMessageResponse with LLM reply and metadata
            
        Raises:
            HTTPException: If processing fails


get_chathistory Responsibilities:
  1. Receive ChatHistoryRequest with user_id, pagination params
        2. Call business logic to fetch from database
        3. Format response
        4. Return ChatHistoryResponse
        
        Args:
            request: ChatHistoryRequest with user_id, session_id, limit, offset
            
        Returns:
            ChatHistoryResponse with messages list and metadata
            
        Raises:
            HTTPException: If retrieval fails
"""