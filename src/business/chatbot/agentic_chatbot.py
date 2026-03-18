# This cahtbot have both vectorDB and SQL database
# SQL Database -> for structured queries (e.g., task management, user profiles)
# VectorDB -> for free-form queries (e.g., knowledge retrieval, document search)
# This chatbot should use the SQL database of each user like last location, last search, etc. 
# to make the conversation more personalized and context-aware. For example, when user ask about the trip suggestion
# it can use the last location of the user to suggest nearby attractions or restaurants.
"""
Agentic chatbot — V3.

Memory architecture:
  Short-term  →  RedisMemory   (recent turns, auto-expires)
  Long-term   →  LongTermMemory / VectorDB  (semantic search over all history)

Tool swap vs V2:
  Removed:  search_chat_history  (SQL LIKE keyword search)
  Added:    search_vector_db     (ChromaDB semantic / embedding search)

After every chat() turn the exchange is persisted to the vector store so
future turns can recall it semantically.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.business.core.prompt_builder import build_agentic_system_prompt
from src.memory.chat_history_manager import ChatHistoryManager
from src.memory.long_term_memory import LongTermMemory
from src.memory.redis_memory import RedisMemory
from src.utils.config import load_config


class AgenticChatbot:
    """
    Conversational agent with semantic long-term memory.

    Dependencies are injected so the chatbot is storage-agnostic and testable.

    Args:
        long_term_memory: LongTermMemory instance (wraps VectorDB + embedder).
        redis_memory:     RedisMemory instance for short-term session context.
        user_id:          Stable identifier for the current user.
        user_info:        Optional dict of user profile data injected into the
                          system prompt (name, location, preferences, …).
        api_key:          OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model_name:       Chat model. Falls back to config → "gpt-4o".
    """

    # ------------------------------------------------------------------ tools
    TOOLS: List[Dict] = [
        {
            "type": "function",
            "function": {
                "name": "search_vector_db",
                "description": (
                    "Semantic search over all past conversations. "
                    "Use this when you need to recall what the user said, "
                    "asked, or was told in a previous session."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language description of what to look for.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # ------------------------------------------------------------ constructor
    def __init__(
        self,
        long_term_memory: LongTermMemory,
        redis_memory: RedisMemory,
        user_id: str,
        chat_history_manager: Optional["ChatHistoryManager"] = None,
        user_info: Optional[Dict] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        load_dotenv()

        self.long_term_memory = long_term_memory
        self.redis_memory = redis_memory
        self.user_id = user_id
        self.chat_history_manager = chat_history_manager
        self.user_info = user_info or {}

        config = load_config()
        llm_config = config.get("llm_config") or {}

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY must be set in environment or passed to __init__")

        self.client = OpenAI(api_key=resolved_key)
        self.model_name = (
            model_name
            or llm_config.get("chat_model")
            or "gpt-4o"
        )

    # ---------------------------------------------------------- tool dispatch
    def _handle_tool_call(self, tool_name: str, tool_args: Dict) -> str:
        if tool_name == "search_vector_db":
            results = self.long_term_memory.recall(
                query=tool_args["query"],
                user_id=self.user_id,
            )
            if not results:
                return "No relevant past conversations found."
            return "\n\n".join(r["text"] for r in results)

        return f"Unknown tool: {tool_name}"

    # ------------------------------------------------------------ agent loop
    def _agent_loop(self, messages: List[Dict]) -> str:
        """
        Runs the ReAct loop: call LLM → handle tool calls → repeat until the
        model returns a plain text response.
        """
        while True:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                # Append assistant message with tool_calls intact
                messages.append(msg)
                for tc in msg.tool_calls:
                    result = self._handle_tool_call(
                        tc.function.name,
                        json.loads(tc.function.arguments),
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                return msg.content

    # --------------------------------------------------------------- chat API
    async def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        chat_summary: Optional[str] = None,
    ) -> str:
        """
        Process one user turn end-to-end.

        1. Fetch recent turns from Redis (short-term window).
        2. Semantic pre-fetch from vector store for likely-relevant context.
        3. Build system prompt (user_info + summary + vector snippets).
        4. Run agent loop (may call search_vector_db on demand).
        5. Persist the exchange to Redis and to the vector store.

        Args:
            user_message: The user's raw input.
            session_id:   Redis key prefix for this session (defaults to user_id).
            chat_summary: Optional rolling summary of the conversation so far.

        Returns:
            The assistant's final response string.
        """
        session_id = session_id or self.user_id

        # 1. Short-term context from Redis (fast, in-session window)
        recent_turns = await self.redis_memory.get_messages(session_id)

        # 1b. If this is a fresh Redis session (no turns yet) and we have a
        #     SQLite manager, pull the previous session's messages as a
        #     rolling summary so the model has prior context on restart.
        if not recent_turns and chat_summary is None and self.chat_history_manager:
            chat_summary = self.chat_history_manager.get_latest_summary(self.user_id)

        # 2. Proactive semantic prefetch — surface relevant long-term memories
        #    before the agent even has a chance to call the tool.
        vector_results = self.long_term_memory.recall(
            query=user_message,
            user_id=self.user_id,
        )

        # 3. Build the system prompt
        system_prompt = build_agentic_system_prompt(
            user_info=self.user_info,
            vector_results=vector_results,
            chat_summary=chat_summary,
        )

        # 4. Assemble message list and run the agent loop
        messages: List[Dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(recent_turns)
        messages.append({"role": "user", "content": user_message})

        response = self._agent_loop(messages)

        # 5a. Persist to short-term (Redis) — fast, auto-expires
        await self.redis_memory.add_message(session_id, "user", user_message)
        await self.redis_memory.add_message(session_id, "assistant", response)

        # 5b. Persist to long-term (vector store) — makes this turn searchable
        self.long_term_memory.remember_conversation(
            user_message=user_message,
            assistant_response=response,
            user_id=self.user_id,
        )

        # 5c. Persist to SQLite — survives Redis TTL expiry and restarts
        if self.chat_history_manager:
            # Use the first user message as the session title
            is_first_turn = not recent_turns
            title = user_message[:60] if is_first_turn else "New conversation"
            self.chat_history_manager.ensure_session(session_id, self.user_id, title)
            self.chat_history_manager.save_message(session_id, self.user_id, "user", user_message)
            self.chat_history_manager.save_message(session_id, self.user_id, "assistant", response)

        return response
