"""
Router Layer - Defines HTTP endpoints and routes to controllers.

In Clean Architecture:
- Router → Defines HTTP endpoints (URLs, methods, DTOs)
- Controller → Handles request/response logic
- Business Logic → Contains processing logic

The router is thin - it just defines routes and delegates to controllers.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from src.database.dto import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatHistoryRequest,
    ChatHistoryResponse
)
from src.api.controller import chat_controller
from src.api.ratelimiter import TokenBucket

router = APIRouter()

# One global bucket shared across all requests (10 req burst, 2 req/sec steady)
_rate_limiter = TokenBucket(capacity=10, refill_rate=2.0)


def _check_rate_limit() -> None:
    if not _rate_limiter.consume(1):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down.",
        )


@router.post("/chat", response_model=ChatMessageResponse, dependencies=[Depends(_check_rate_limit)])
async def chat_endpoint(request: ChatMessageRequest):
    """
    Chat endpoint that accepts free-form text.
    
    Flow:
    1. FastAPI validates request against ChatMessageRequest DTO
    2. Router delegates to controller
    3. Controller handles business logic and error handling
    4. Returns ChatMessageResponse
    
    The DTO (ChatMessageRequest) ensures:
    - The request body has the correct structure
    - 'message' field exists and is a string (but can be ANY text)
    - Optional fields like user_id, session_id are validated
    - FastAPI auto-generates OpenAPI docs
    """
    # Delegate to controller - keeps router thin
    return await chat_controller.send_message(request)


@router.get("/history", response_model=ChatHistoryResponse)
async def chat_history_endpoint(
    user_id: int,
    session_id: str = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Retrieve chat history for a user.
    
    Flow:
    1. FastAPI extracts query parameters
    2. Router creates ChatHistoryRequest DTO
    3. Router delegates to controller
    4. Returns ChatHistoryResponse
    """
    # Convert query params to DTO
    request = ChatHistoryRequest(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        offset=offset
    )
    
    # Delegate to controller
    return await chat_controller.get_chat_history(request)


# Example showing the difference:
# WITHOUT DTO (bad practice):
# @router.post("/chat")
# async def chat_bad(request: dict):  # No validation, no docs, no type safety
#     message = request.get("message")  # Could be missing, wrong type, etc.
#     ...

# WITH DTO (good practice):
# @router.post("/chat")
# async def chat_good(request: ChatMessageRequest):  # Validated, documented, type-safe
#     message = request.message  # Guaranteed to exist and be a string
#     ...

