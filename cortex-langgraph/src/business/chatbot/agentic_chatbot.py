"""
Agentic chatbot — LangGraph edition.

Memory architecture (unchanged from cortex-core):
  Short-term  →  RedisMemory     (recent turns, auto-expires)
  Long-term   →  LongTermMemory  (semantic search over all history, Chroma)
  Persistent  →  ChatHistoryManager (SQLite, survives restarts; feeds the history API)

What changed: the tool-calling loop.

cortex-core ran a hand-written `while True:` loop — call the model, if it asked
for tools run them and append the results, repeat until it answers in plain
text. Here that loop is a LangGraph `StateGraph` with two nodes:

        ┌─────────┐   tools_condition: model asked for tools?   ┌─────────┐
  START │  agent  │ ───────────────────── yes ────────────────▶ │  tools  │
  ────▶ │ (LLM)   │ ◀──────────────────────────────────────────  │ToolNode │
        └────┬────┘                                             └─────────┘
             │ no (plain answer)
             ▼
            END

- `agent`  calls the chat model with the tools bound (`llm.bind_tools`).
- `tools`  is LangGraph's prebuilt `ToolNode`: it runs every tool call in the
           last AI message and appends one ToolMessage per call.
- `tools_condition` is the prebuilt router that picks between the two.

The loop is the same; what the graph adds is that it's inspectable (you can
draw it, stream each step, trace it) and bounded — LangGraph stops a runaway
loop at `recursion_limit` (see MAX_TOOL_ROUNDS) instead of spinning forever.

Why short-term memory is NOT a LangGraph checkpointer: the official Redis
checkpointer needs Redis with the JSON and Search modules (Redis Stack / Redis 8
with the query engine). Plain Redis — what Docker Compose, Homebrew and Azure
Cache for Redis Basic/Standard provide — doesn't have them. So conversation
turns stay in plain Redis lists, and are converted to LangChain messages when
they enter the graph.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, convert_to_messages
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.business.chatbot.tools import build_tools
from src.business.core.live_data import LiveDataProvider, create_live_data_provider
from src.business.core.model import create_llm, model_label, resolve_provider
from src.business.core.prompt_builder import build_agentic_system_prompt
from src.memory.chat_history_manager import ChatHistoryManager
from src.memory.long_term_memory import LongTermMemory
from src.memory.redis_memory import ShortTermMemoryBase
from src.utils.config import load_config

# Providers whose chat models support OpenAI-style tool calling.
TOOL_CALLING_PROVIDERS = ("openai", "azure_openai")

# Upper bound on agent → tools round-trips per user turn. cortex-core's
# while-loop had no cap at all, and LangGraph's own default (10,007 steps) is
# effectively none either. Each round is two graph steps (agent, then tools);
# the final answer needs two more (the answering agent step, and the step that
# reaches END). Exceeding it raises GraphRecursionError.
# It means LangGraph puts a safety limit on how many times the agent can call tools in one user request.
#If you configure max_tool_rounds = 5, the agent can make at most 5 tool-call rounds.
#If it exceeds the limit, LangGraph raises GraphRecursionError.

MAX_TOOL_ROUNDS = int((load_config().get("agent") or {}).get("max_tool_rounds", 10))  # config.yml → agent.max_tool_rounds
RECURSION_LIMIT = 2 * MAX_TOOL_ROUNDS + 2


class AgenticChatbot:
    """
    Conversational agent with semantic long-term memory.

    Dependencies are injected so the chatbot is storage-agnostic and testable.

    Args:
        long_term_memory: LongTermMemory instance (wraps a LangChain vector store).
        redis_memory:     Short-term session memory (RedisMemory in production).
        user_id:          Stable identifier for the current user.
        chat_history_manager: Optional SQLite history (persistent, feeds the API).
        user_info:        Optional dict of user profile data injected into the
                          system prompt (name, location, preferences, …).
        api_key:          Provider API key. Falls back to the provider's env var.
        model_name:       Override for the chat model / Azure deployment. Defaults to
                          config.yml → models.chat.name.
        live_data_provider: Backend for the web_search tool.
        llm:              A ready chat model. Injected by tests; when omitted the
                          model is built from LLM_PROVIDER via create_llm().
    """

    def __init__(
        self,
        long_term_memory: LongTermMemory,
        redis_memory: ShortTermMemoryBase,
        user_id: str,
        chat_history_manager: Optional[ChatHistoryManager] = None,
        user_info: Optional[Dict] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        live_data_provider: Optional[LiveDataProvider] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        load_dotenv()

        self.long_term_memory = long_term_memory
        self.redis_memory = redis_memory
        self.user_id = user_id
        self.chat_history_manager = chat_history_manager
        self.user_info = user_info or {}
        self.live_data_provider = live_data_provider or create_live_data_provider()
        self.last_usage: Dict[str, int] = {}

        if llm is None:
            llm = self._build_llm(api_key=api_key, model_name=model_name)
        self.llm = llm
        self.model_name = model_label(llm)

        self.tools = build_tools(
            long_term_memory=self.long_term_memory,
            live_data_provider=self.live_data_provider,
            user_id=self.user_id,
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------- the model
    @staticmethod
    def _build_llm(api_key: Optional[str], model_name: Optional[str]) -> BaseChatModel:
        # LLM_PROVIDER selects the model, the same switch used by create_llm()
        # everywhere else. Local Hugging Face models don't support this agent's
        # tool-calling flow, so that provider isn't valid here — fail loudly
        # rather than silently answering without tools.
        provider = resolve_provider()
        if provider not in TOOL_CALLING_PROVIDERS:
            raise NotImplementedError(
                f"AgenticChatbot's tool-calling graph only supports LLM_PROVIDER in "
                f"{TOOL_CALLING_PROVIDERS} — got {provider!r}. Local Hugging Face "
                "models don't support this agent's function-calling flow."
            )

        # role="chat": the model comes from config.yml → models.chat (for
        # azure_openai, from AZURE_OPENAI_CHAT_DEPLOYMENT_NAME). An explicit
        # model_name still overrides it.
        return create_llm(provider, role="chat", model_name=model_name, api_key=api_key)

    # ------------------------------------------------------------- the graph
    def _build_graph(self):
        """Compile the agent ⇄ tools graph (replaces cortex-core's while-loop)."""
        llm_with_tools = self.llm.bind_tools(self.tools)

        def agent(state: MessagesState) -> Dict[str, List[BaseMessage]]:
            # MessagesState's reducer APPENDS returned messages to the state,
            # so each step only returns what's new.
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        builder = StateGraph(MessagesState)
        builder.add_node("agent", agent)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)  # → "tools" or END
        builder.add_edge("tools", "agent")
        return builder.compile().with_config(recursion_limit=RECURSION_LIMIT)

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
        4. Run the agent graph (may call search_vector_db / web_search on demand).
        5. Persist the exchange to Redis, the vector store, and SQLite.

        Args:
            user_message: The user's raw input.
            session_id:   Redis key prefix for this session (defaults to user_id).
            chat_summary: Optional rolling summary of the conversation so far.

        Returns:
            The assistant's final response string. Token usage for the turn
            (summed over every model call the graph made) is left on
            `self.last_usage`.
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

        # 4. Run the graph. Redis stores plain {"role", "content"} dicts;
        #    convert_to_messages turns them into Human/AI messages.
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        messages.extend(convert_to_messages(recent_turns))
        messages.append(HumanMessage(content=user_message))

        result = await self.graph.ainvoke({"messages": messages})
        new_messages = result["messages"][len(messages):]
        response = self._final_text(new_messages)
        self.last_usage = self._sum_usage(new_messages)

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

    # ---------------------------------------------------------------- helpers
    #These are two helper functions that process the messages produced by the LangGraph run.
    # extract the final answer from the model.
    @staticmethod
    def _final_text(new_messages: List[BaseMessage]) -> str:
        """The graph ends on the model's plain-text answer — the last AIMessage."""
        for message in reversed(new_messages):
            if isinstance(message, AIMessage):
                # .text flattens both plain-string and content-block responses.
                return message.text
        return ""

    @staticmethod
    #Purpose: calculate the total token usage for the entire user turn.
    def _sum_usage(new_messages: List[BaseMessage]) -> Dict[str, int]:
        """Add up token usage across every model call made during this turn.

        A turn with two tool rounds calls the model three times; cortex-core
        never reported tokens at all, so cost tracking could never fire.
        """
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for message in new_messages:
            usage = getattr(message, "usage_metadata", None)
            if usage:
                for key in totals:
                    totals[key] += int(usage.get(key, 0) or 0)
        return totals


# So these functions are not LangGraph orchestration. 
# They are application-level helpers used after/during 
# the graph execution to extract the result and track usage/cost.