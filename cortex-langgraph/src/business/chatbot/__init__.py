"""
Business logic entry-points for the chat API.

process_chat_message  →  called by ChatController.send_message
get_chat_history      →  called by ChatController.get_chat_history
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from src.business.chatbot.agentic_chatbot import AgenticChatbot
from src.database.dto import ChatMessageRequest, ChatHistoryRequest
from src.memory.chat_history_manager import ChatHistoryManager
from src.memory.long_term_memory import LongTermMemory
from src.memory.redis_memory import create_memory
from src.memory.vectordb import create_conversation_vector_store
from src.business.core.embedding import create_embedder
from src.utils.config import load_config

load_dotenv()

# Project root = src/business/chatbot/__init__.py → up 3 levels
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(relative_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(relative_path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Dependency factory
# ---------------------------------------------------------------------------
def _make_chatbot(user_id: str) -> AgenticChatbot:
    cfg = load_config()
    dirs = cfg.get("directories", {})

    api_key = os.getenv("OPENAI_API_KEY")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Chroma persistent path — anchored to project root
    vectordb_dir = _resolve(os.getenv("VECTORDB_DIR") or dirs.get("vectordb_dir", "data/vectordb"))
    vectordb_dir.mkdir(parents=True, exist_ok=True)

    # SQLite path — anchored to project root
    db_path = _resolve(os.getenv("SQLITE_DB_PATH") or dirs.get("db_path", "data/chatbot.db"))

    # The vector store embeds text itself, so the embedder is handed to it
    # rather than to LongTermMemory (which is where cortex-core put it).
    embedder = create_embedder(api_key=api_key)
    vectorstore = create_conversation_vector_store(
        collection_name="chat_history",
        embedding=embedder,
        persist_directory=str(vectordb_dir),
    )
    long_term_memory = LongTermMemory(vectorstore=vectorstore)
    redis_memory = create_memory(url=redis_url)
    chat_history_manager = ChatHistoryManager(db_path=str(db_path))

    return AgenticChatbot(
        long_term_memory=long_term_memory,
        redis_memory=redis_memory,
        user_id=user_id,
        chat_history_manager=chat_history_manager,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def process_chat_message(request: ChatMessageRequest) -> Dict:
    user_id = str(request.user_id) if request.user_id else "anonymous"
    session_id = request.session_id or user_id

    chatbot = _make_chatbot(user_id)
    reply = await chatbot.chat(
        user_message=request.message,
        session_id=session_id,
    )

    # LangChain reports token usage on every model response, so — unlike
    # cortex-core, which always returned tokens_used=None — the controller now
    # receives real counts and its cost-tracking branch actually runs.
    usage = chatbot.last_usage
    return {
        "reply": reply,
        "model_used": chatbot.model_name,
        "tokens_used": usage.get("total_tokens") or None,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


async def get_chat_history(request: ChatHistoryRequest) -> Dict:
    """
    Returns messages for a session from SQLite — persistent across restarts.
    Falls back to an empty list gracefully if the session doesn't exist yet.
    """
    cfg = load_config()
    db_path = _resolve(
        os.getenv("SQLITE_DB_PATH")
        or cfg.get("directories", {}).get("db_path", "data/chatbot.db")
    )
    manager = ChatHistoryManager(db_path=str(db_path))

    session_id = request.session_id or str(request.user_id)
    messages = manager.get_messages(
        session_id=session_id,
        limit=request.limit,
        offset=request.offset,
    )
    total = manager.count_messages(session_id)

    return {
        "messages": messages,
        "total": total,
        "session_id": session_id,
    }
