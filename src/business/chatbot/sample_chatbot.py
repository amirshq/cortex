# Here is the chatbot implementation

"""
Example: How auto-detection works for StructuredQueryRequest vs FreeFormQueryRequest

The user's understanding is CORRECT for auto-detection approach:
- Backend receives free-form text
- Backend checks/parses the format
- Backend routes to structured or free-form handler
"""

from src.database.dto import ChatMessageRequest, StructuredQueryRequest, FreeFormQueryRequest
from src.business.rag.retrieval import RAGPipeline
from pathlib import Path
import os


def detect_query_type(message: str) -> str:
    """
    Detects if user input is command-style or natural language.
    
    In practice, you'd use:
    - LLM with function calling (OpenAI, Hugging Face)
    - Intent classification model
    - Rule-based parsing
    - Or a combination
    """
    message_lower = message.lower().strip()
    
    # Simple rule-based detection (in production, use LLM)
    command_keywords = ["create", "delete", "update", "list", "show", "get"]
    action_patterns = ["create task", "list tasks", "delete task"]
    
    # Check if it looks like a command
    if any(keyword in message_lower for keyword in command_keywords):
        if any(pattern in message_lower for pattern in action_patterns):
            return "structured"  # Command-style
    
    return "freeform"  # Natural language conversation


def parse_to_structured(message: str) -> StructuredQueryRequest:
    """
    Parses free-form text into structured command.
    
    Example:
    Input: "Create a task to buy groceries tomorrow at 3pm"
    Output: StructuredQueryRequest(
        action="create_task",
        parameters={
            "title": "buy groceries",
            "due_date": "tomorrow",
            "time": "3pm"
        }
    )
    """
    # In production, use LLM with function calling or structured output
    # This is a simplified example
    
    message_lower = message.lower()
    
    if "create task" in message_lower or "create a task" in message_lower:
        # Extract task details (simplified - use LLM in production)
        return StructuredQueryRequest(
            query_type="structured",
            action="create_task",
            parameters={
                "title": message,  # In production, extract with LLM
                "raw_message": message
            }
        )
    
    # Add more parsing logic for other actions
    raise ValueError(f"Cannot parse message as structured: {message}")


async def process_chat_message(request: ChatMessageRequest):
    """
    Main handler that auto-detects query type and routes accordingly.
    
    This is what you described - the backend checks the format
    and decides whether to use StructuredQueryRequest or FreeFormQueryRequest.
    """
    message = request.message
    
    # Step 1: Detect query type
    query_type = detect_query_type(message)
    
    # Step 2: Route based on detection
    if query_type == "structured":
        # Parse to structured format
        structured_request = parse_to_structured(message)
        
        # Process as structured command
        return await handle_structured_query(structured_request)
    
    else:
        # Treat as free-form conversation
        freeform_request = FreeFormQueryRequest(
            query_type="freeform",
            message=message
        )
        
        # Process as natural language conversation
        return await handle_freeform_query(freeform_request)


async def handle_structured_query(request: StructuredQueryRequest):
    """Handle structured commands (e.g., create_task, list_tasks)."""
    if request.action == "create_task":
        # Execute task creation logic
        # Use request.parameters to get task details
        return {"status": "task_created", "action": request.action}
    
    # Handle other actions...
    return {"status": "processed", "action": request.action}


async def handle_freeform_query(request: FreeFormQueryRequest):
    """Handle free-form natural language conversation via RAG."""
    persist_dir = Path(os.getenv("VECTOR_STORE_DIR", "src/business/rag/vectorstore"))
    rag = RAGPipeline(persist_dir=str(persist_dir))
    answer, confidence, _ = rag.answer(request.message)
    return {"reply": answer, "type": "freeform", "confidence": confidence}


# Example usage flow:
"""
1. User sends: "Create a task to buy groceries tomorrow"
   ↓
2. Backend receives: ChatMessageRequest(message="Create a task...")
   ↓
3. Backend detects: detect_query_type() → returns "structured"
   ↓
4. Backend parses: parse_to_structured() → StructuredQueryRequest
   ↓
5. Backend processes: handle_structured_query() → executes create_task
   ↓
6. Response sent to user

OR

1. User sends: "How are you today?"
   ↓
2. Backend receives: ChatMessageRequest(message="How are you today?")
   ↓
3. Backend detects: detect_query_type() → returns "freeform"
   ↓
4. Backend processes: handle_freeform_query() → sends to LLM
   ↓
5. Response sent to user
"""
