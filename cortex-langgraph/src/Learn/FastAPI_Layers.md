# FastAPI Architecture Layers

This document explains the complete layered architecture used in FastAPI applications, following enterprise-grade **N-Tier** or **Multilayered Architecture** patterns.

![FastAPI Layers Architecture](../Images/FastAPI%20Layers.png)

---

## 📐 Complete Architecture Overview

In a standard enterprise architecture, there are several layers that facilitate communication, data integrity, and storage. Here is the complete flow from client to database:

---

## 🔄 Layer-by-Layer Breakdown

### 1. **Client Layer (Presentation Layer)**

**Location:** Frontend application (React, mobile app, CLI, etc.)

**Responsibility:**
- User interface and interaction
- Sends HTTP requests to the API
- Displays responses to users

**Example:**
```typescript
// React frontend
const response = await fetch('/api/v1/chat', {
  method: 'POST',
  body: JSON.stringify({ message: "Hello" })
});
```

---

### 2. **Middleware / Security Layer**

**Location:** `src/api/middleware/` or FastAPI middleware

**Responsibility:**
- **Authentication**: JWT token verification
- **Rate Limiting**: Prevents API abuse
- **CORS**: Cross-Origin Resource Sharing handling
- **Logging**: Request/response logging
- **Error Handling**: Global exception handling

**Example:**
```python
# Middleware runs before Router
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Verify JWT token
    # Check rate limits
    # Log request
    response = await call_next(request)
    return response
```

---

### 3. **Router Layer**

**Location:** `src/api/router.py`

**Responsibility:**
- Entry point of the server
- Parses incoming URL and HTTP method (GET, POST, PUT, DELETE)
- Routes requests to specific controller methods
- Defines API endpoints and paths

**Example:**
```python
from fastapi import APIRouter
from src.api.controller import chat_controller

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(request: ChatMessageRequest):
    return await chat_controller.send_message(request)
```

---

### 4. **Controller Layer**

**Location:** `src/api/controller.py`

**Responsibility:**
- Orchestrates request handling
- Extracts data from request (params, body, query)
- Validates basic format and business rules
- Calls business logic layer
- Formats responses
- Handles errors and converts to HTTP responses

**Example:**
```python
class ChatController:
    @staticmethod
    async def send_message(request: ChatMessageRequest) -> ChatMessageResponse:
        # Validate business rules
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Call business logic
        result = await process_chat_message(request)
        
        # Format response
        return ChatMessageResponse(reply=result["reply"])
```

---

### 5. **Data Transfer Object (DTO) / Validation Layer**

**Location:** `src/data/dto.py`

**Responsibility:**
- Defines request/response data structures
- Validates data types and required fields
- Ensures data schema compliance
- Provides type safety
- Auto-generates API documentation (OpenAPI/Swagger)

**Example:**
```python
from pydantic import BaseModel, Field

class ChatMessageRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: Optional[int] = None
    session_id: Optional[str] = None
```

---

### 6. **Business Logic Layer (Service Layer)**

**Location:** `src/business/chatbot.py`

**Responsibility:**
- Contains core business rules and logic
- Processes data according to business requirements
- Independent of database or transport protocol
- Handles complex operations (LLM calls, calculations, etc.)
- Can be tested independently

**Example:**
```python
async def process_chat_message(request: ChatMessageRequest):
    # Detect query type
    query_type = detect_query_type(request.message)
    
    # Route to appropriate handler
    if query_type == "structured":
        return await handle_structured_query(request)
    else:
        return await handle_freeform_query(request)
```

---

### 7. **Data Access Layer (Repository Layer)**

**Location:** `src/data/repositories/` or `src/infrastructure/db/repositories/`

**Responsibility:**
- Abstracts database operations
- Provides methods like `find_by_id()`, `save()`, `delete()`
- Allows switching databases without breaking business logic
- Encapsulates SQL queries
- Handles database transactions

**Example:**
```python
class ChatRepository:
    async def save_message(self, message: Message) -> Message:
        async with self.session.begin():
            self.session.add(message)
            return message
    
    async def get_history(self, user_id: int, limit: int) -> List[Message]:
        return await self.session.query(Message)\
            .filter(Message.user_id == user_id)\
            .limit(limit)\
            .all()
```

---

### 8. **Persistence / Database Layer**

**Location:** Database (PostgreSQL, MongoDB, SQLite, etc.)

**Responsibility:**
- Physical storage of data
- Data persistence
- ACID transactions
- Data integrity

**Example:**
```python
# SQLAlchemy models
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    content = Column(String)
    user_id = Column(Integer)
```

---

## 🔄 Complete Data Flow Example

Let's trace a request through all layers:

```
1. Client Layer
   User types: "Create a task to buy groceries"
   ↓ HTTP POST /api/v1/chat
   
2. Middleware Layer
   - Verifies JWT token ✓
   - Checks rate limit ✓
   - Logs request
   ↓
   
3. Router Layer
   @router.post("/chat") → routes to chat_endpoint()
   ↓
   
4. Controller Layer
   ChatController.send_message()
   - Validates: message not empty ✓
   - Extracts: request.message
   ↓
   
5. DTO Layer
   ChatMessageRequest validates:
   - message is string ✓
   - user_id is optional int ✓
   ↓
   
6. Business Logic Layer
   process_chat_message()
   - Detects: "structured" query
   - Parses: action="create_task"
   - Processes: Creates task
   ↓
   
7. Repository Layer
   TaskRepository.save()
   - Builds SQL query
   - Executes transaction
   ↓
   
8. Database Layer
   INSERT INTO tasks (...) VALUES (...)
   - Stores data
   - Returns result
   ↑
   
7. Repository Layer ← Receives data
   ↑
   
6. Business Logic Layer ← Formats result
   ↑
   
5. DTO Layer ← Converts to ChatMessageResponse
   ↑
   
4. Controller Layer ← Returns response
   ↑
   
3. Router Layer ← HTTP 200 OK
   ↑
   
2. Middleware Layer ← Adds headers
   ↑
   
1. Client Layer ← Displays: "Task created successfully!"
```

---

## 📊 Layer Summary Table

| Layer | Location | Primary Responsibility | Key Technologies |
|-------|----------|----------------------|------------------|
| **Client** | Frontend | User Interface & Interaction | React, Vue, Mobile Apps |
| **Middleware** | `src/api/middleware/` | Security, Auth, Rate Limiting | FastAPI Middleware, JWT |
| **Router** | `src/api/router.py` | URL Mapping & Request Routing | FastAPI Router |
| **Controller** | `src/api/controller.py` | Request/Response Handling | FastAPI, HTTPException |
| **DTO** | `src/data/dto.py` | Data Validation & Schema | Pydantic |
| **Business Logic** | `src/business/` | Core Business Rules | Python, LLM APIs |
| **Repository** | `src/data/repositories/` | Data Querying & Persistence | SQLAlchemy, Repository Pattern |
| **Database** | Database Server | Physical Storage | PostgreSQL, MongoDB, SQLite |

---

## 🎯 Key Principles

### 1. **Separation of Concerns**
Each layer has a single, well-defined responsibility.

### 2. **Dependency Direction**
Dependencies flow downward:
- Client → Middleware → Router → Controller → Business Logic → Repository → Database

### 3. **Independence**
- Business logic is independent of database
- Controllers are independent of business logic implementation
- Easy to test each layer in isolation

### 4. **Abstraction**
- Repository abstracts database details
- DTOs abstract data structure
- Controllers abstract HTTP details

---

## 🧪 Testing Each Layer

```python
# Test Router
def test_router():
    response = client.post("/api/v1/chat", json={"message": "Hello"})
    assert response.status_code == 200

# Test Controller
async def test_controller():
    request = ChatMessageRequest(message="Hello")
    response = await chat_controller.send_message(request)
    assert response.reply is not None

# Test Business Logic
async def test_business_logic():
    request = ChatMessageRequest(message="Hello")
    result = await process_chat_message(request)
    assert "reply" in result

# Test Repository
async def test_repository():
    message = await chat_repository.save_message(Message(...))
    assert message.id is not None
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Clean Architecture Principles](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

## ✅ Benefits of This Architecture

1. **Maintainability**: Easy to modify one layer without affecting others
2. **Testability**: Each layer can be tested independently
3. **Scalability**: Can scale different layers independently
4. **Flexibility**: Easy to swap implementations (e.g., change database)
5. **Team Collaboration**: Different teams can work on different layers
6. **Code Reusability**: Business logic can be reused across different interfaces

---

This layered architecture ensures your FastAPI application is robust, maintainable, and follows industry best practices! 🚀
