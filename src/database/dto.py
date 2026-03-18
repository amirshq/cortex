# RECOMMENDED APPROACH for chatbot and how Chatbot use the dto.py:
# 1. Client always sends: ChatMessageRequest (free-form text)
# 2. Backend receives: ChatMessageRequest
# 3. Backend uses LLM/parser to detect intent:
#    - If the user message is like command-like → convert to StructuredQueryRequest internally
#    - If the user message is like conversational → treat as FreeFormQueryRequest internally
# 4. Backend processes accordingly


"""
DTOs (Data Transfer Objects) for the chatbot API.

Even though the user's query text is free-form and unstructured,
we still need DTOs to:
1. Define the API contract (request/response structure)
2. Validate required fields and data types
3. Handle metadata (user_id, session_id, timestamps, etc.)
4. Enable API documentation (OpenAPI/Swagger)
5. Type safety and IDE support
"""



from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatMessageRequest(BaseModel):
    """
    Request DTO for chat messages.
    
    The 'message' field accepts ANY free-form text - no constraints on format.
    But we still need this DTO to:
    - Structure the HTTP request body
    - Validate that 'message' is provided and is a string
    - Include metadata like user_id, session_id, etc.
    """
    message: str = Field(..., description="Free-form user query - any format accepted")
    user_id: Optional[int] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Chat session identifier")
    context: Optional[dict] = Field(None, description="Additional context/metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What's the weather like today?",
                "user_id": 1,
                "session_id": "abc123",
                "context": {"timezone": "UTC"}
            }
        }


class ChatMessageResponse(BaseModel):
    """
    Response DTO for chat messages.
    
    Even though the LLM response is free-form text, we structure it
    with metadata for better API design.
    """
    reply: str = Field(..., description="Free-form LLM response - any format")
    session_id: Optional[str] = Field(None, description="Chat session identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_used: Optional[str] = Field(None, description="LLM model identifier")
    tokens_used: Optional[int] = Field(None, description="Token count for this response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "reply": "The weather is sunny and 72°F today.",
                "session_id": "abc123",
                "timestamp": "2024-01-15T10:30:00",
                "model_used": "zephyr-7b-beta",
                "tokens_used": 45
            }
        }


class ChatHistoryRequest(BaseModel):
    """DTO for retrieving chat history - structured query parameters."""
    # this class request format for retrieving chat history from Database (SQLAlchemy models) to API Client
    user_id: int
    session_id: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ChatHistoryResponse(BaseModel):
    """DTO for chat history response."""
    # this class response format for retrieving chat history from Database (SQLAlchemy models) to API Client
    messages: List[dict] = Field(..., description="List of message objects")
    total: int = Field(..., description="Total number of messages")
    session_id: Optional[str] = None


# Alternative: If you want to support multiple input formats, use discriminated unions
class StructuredQueryRequest(BaseModel):
    """
    For structured queries (e.g., task creation).
    
    NOTE: This is typically used INTERNALLY by the backend after parsing.
    The client usually sends free-form text via ChatMessageRequest,
    and the backend's LLM/parser determines if it should be treated as
    a structured command (like "create_task") or free-form conversation.
    
    Example of what this represents AFTER parsing:
    User says: "Create a task to buy groceries tomorrow"
    → Backend parses to: {"action": "create_task", "parameters": {...}}


    User types: "Create a task to buy groceries tomorrow"
         ↓
    Backend receives: ChatMessageRequest (free-form text)
         ↓
    Backend checks format:
        - Looks like command? → Use StructuredQueryRequest
        - Looks like conversation? → Use FreeFormQueryRequest
         ↓
Backend processes accordingly
    """
    query_type: str = "structured"
    action: str  # "create_task", "list_tasks", etc.
    parameters: dict


class FreeFormQueryRequest(BaseModel):
    """
    For free-form natural language queries.
    
    NOTE: This is typically used INTERNALLY by the backend after parsing.
    The client sends free-form text, and the backend determines if it's
    a simple conversation (freeform) or needs structured action (structured).
    
    Example: User says "How are you?" → treated as freeform conversation
    """
    query_type: str = "freeform"
    message: str


# Union type for multiple input formats (if needed)
# This would be used if you want to accept BOTH formats from the client
from typing import Union
ChatRequest = Union[StructuredQueryRequest, FreeFormQueryRequest, ChatMessageRequest]


# ---------------------------------------------------------------------------
# RAG DTOs
# ---------------------------------------------------------------------------

class RAGQueryRequest(BaseModel):
    """Request DTO for RAG pipeline queries."""
    question: str = Field(..., description="Natural-language question to answer from uploaded PDFs")

    class Config:
        json_schema_extra = {
            "example": {"question": "What are the main findings in the document?"}
        }


class RAGQueryResponse(BaseModel):
    """Response DTO for RAG pipeline answers."""
    answer: str = Field(..., description="Generated answer from the RAG pipeline")
    sources: List[dict] = Field(default_factory=list, description="Relevant source chunks used")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The main findings are...",
                "sources": [{"text": "...", "metadata": {}, "score": 0.87}],
            }
        }


class RAGUploadResponse(BaseModel):
    """Response DTO after uploading and indexing a PDF."""
    filename: str
    docs_indexed: int
    chunks_indexed: int
    message: str = "PDF indexed successfully"



