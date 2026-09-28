# Step-by-Step Guide: Understanding `controller.py`

## 🎯 What is a Controller?

A **controller** is a layer in clean architecture that:
- Receives HTTP requests (via DTOs)
- Validates input
- Calls business logic
- Handles errors
- Returns HTTP responses (via DTOs)

Think of it as a **bridge** between the HTTP layer (router) and business logic.

---

## 📐 Architecture Flow

```
Client (Frontend)
    ↓ HTTP Request
Router (router.py) → Defines endpoints
    ↓
Controller (controller.py) → Handles request/response
    ↓
Business Logic (business/chatbot.py) → Processes data
    ↓
Database/LLM → Stores/Generates data
    ↓
Controller ← Formats response
    ↓
Router ← Returns HTTP response
    ↓
Client ← Receives JSON
```

---

## 📝 Step-by-Step: How `controller.py` Works

### Step 1: Import Dependencies

```python
from fastapi import HTTPException, status
from src.data.dto import ChatMessageRequest, ChatMessageResponse
from src.business.chatbot import process_chat_message
```

**What this does:**
- `HTTPException`: For returning HTTP error responses
- `status`: HTTP status codes (400, 500, etc.)
- DTOs: Request/response data structures
- Business logic: The actual processing functions

---

### Step 2: Create Controller Class

```python
class ChatController:
    """Controller for chat-related endpoints."""
```

**Why a class?**
- Organizes related methods
- Can hold state if needed
- Easy to test
- Can be instantiated as singleton

---

### Step 3: Create `send_message` Method

```python
@staticmethod
async def send_message(request: ChatMessageRequest) -> ChatMessageResponse:
```

**Breaking it down:**

#### 3.1: Method Signature
- `@staticmethod`: Doesn't need instance (can call directly)
- `async`: Handles async operations (database, LLM calls)
- `request: ChatMessageRequest`: Input DTO (validated by FastAPI)
- `-> ChatMessageResponse`: Output DTO (what we return)

#### 3.2: Validation
```python
if not request.message or not request.message.strip():
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Message cannot be empty"
    )
```

**What this does:**
- DTO already validates structure (message is string)
- Controller validates business rules (message not empty)
- Raises HTTP 400 if invalid

#### 3.3: Call Business Logic
```python
result = await process_chat_message(request)
```

**What this does:**
- Calls the business logic function
- `await` because it's async (might call LLM, database)
- Returns result from business layer

#### 3.4: Format Response
```python
return ChatMessageResponse(
    reply=result.get("reply", "I'm sorry, I couldn't process that."),
    session_id=request.session_id,
    model_used=result.get("model_used", "zephyr-7b-beta"),
    tokens_used=result.get("tokens_used")
)
```

**What this does:**
- Converts business logic result to response DTO
- Extracts data from result dict
- Provides defaults if missing
- Returns structured response

#### 3.5: Error Handling
```python
except ValueError as e:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(e)
    )
except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Internal server error: {str(e)}"
    )
```

**What this does:**
- Catches validation errors → HTTP 400
- Catches unexpected errors → HTTP 500
- Converts exceptions to HTTP responses
- Provides error messages to client

---

### Step 4: Create `get_chat_history` Method

Similar pattern:
1. Validate input
2. Call business logic
3. Format response
4. Handle errors

---

### Step 5: Create Singleton Instance

```python
chat_controller = ChatController()
```

**Why?**
- Easy to import and use
- Single instance (no need to create new each time)
- Used in router: `chat_controller.send_message(request)`

---

## 🔄 Complete Request Flow Example

### Example: User sends "Hello"

```
1. Client sends POST /api/v1/chat
   {
     "message": "Hello",
     "user_id": 1,
     "session_id": "abc123"
   }

2. Router (router.py) receives request
   @router.post("/chat")
   async def chat_endpoint(request: ChatMessageRequest):
       return await chat_controller.send_message(request)

3. Controller (controller.py) processes
   - Validates: message not empty ✓
   - Calls: process_chat_message(request)
   - Business logic: Detects "freeform", sends to LLM
   - Formats: ChatMessageResponse(reply="Hi there!", ...)
   - Returns: Response DTO

4. Router returns HTTP response
   {
     "reply": "Hi there!",
     "session_id": "abc123",
     "timestamp": "2024-01-15T10:00:00",
     "model_used": "zephyr-7b-beta"
   }

5. Client receives JSON response
```

---

## 🎓 Key Concepts to Remember

### 1. **Separation of Concerns**
- **Router**: Defines routes only
- **Controller**: Handles request/response logic
- **Business Logic**: Contains processing logic
- **DTOs**: Define data structures

### 2. **Error Handling**
- Always wrap business logic in try/except
- Convert exceptions to HTTPException
- Provide meaningful error messages

### 3. **Validation**
- DTOs validate structure (types, required fields)
- Controller validates business rules (not empty, valid ranges, etc.)

### 4. **Async/Await**
- Use `async` for methods that call async functions
- Use `await` when calling async business logic
- FastAPI handles async automatically

---

## 🛠️ How to Add a New Endpoint

### Step 1: Add DTO (if needed)
```python
# In dto.py
class NewRequest(BaseModel):
    field: str
```

### Step 2: Add Business Logic
```python
# In business/chatbot.py
async def new_function(request: NewRequest):
    # Process logic
    return {"result": "data"}
```

### Step 3: Add Controller Method
```python
# In controller.py
@staticmethod
async def new_endpoint(request: NewRequest) -> NewResponse:
    try:
        result = await new_function(request)
        return NewResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 4: Add Router Endpoint
```python
# In router.py
@router.post("/new")
async def new_route(request: NewRequest):
    return await chat_controller.new_endpoint(request)
```

---

## ✅ Best Practices

1. **Keep controllers thin**: Delegate to business logic
2. **Handle all errors**: Never let exceptions bubble up unhandled
3. **Use DTOs**: Always use DTOs for request/response
4. **Document methods**: Add docstrings explaining what each method does
5. **Validate early**: Check business rules in controller
6. **Return proper HTTP codes**: 400 for bad requests, 500 for server errors

---

## 🧪 Testing the Controller

You can test the controller independently:

```python
# test_controller.py
from src.api.controller import chat_controller
from src.data.dto import ChatMessageRequest

async def test_send_message():
    request = ChatMessageRequest(
        message="Hello",
        user_id=1
    )
    response = await chat_controller.send_message(request)
    assert response.reply is not None
    assert response.session_id is not None
```

---

## 📚 Summary

**Controller = Request Handler + Error Handler + Response Formatter**

1. Receives validated DTO from router
2. Validates business rules
3. Calls business logic
4. Formats response as DTO
5. Handles errors as HTTP responses

The controller is the **glue** between HTTP and business logic! 🎯

