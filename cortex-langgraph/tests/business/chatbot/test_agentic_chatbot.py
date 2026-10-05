"""Tests for AgenticChatbot — the LangGraph agent and the memory fan-out.

This is the orchestration layer: the code that decides which tools run,
how many times the model is called, and what gets persisted where. None
of it is exercised by the controller tests (which mock
`process_chat_message` wholesale).

Everything here is hermetic — the chat model is a scripted FakeChatModel,
and Redis / long-term memory / SQLite are in-memory fakes.

Ported from cortex-core's tests of the hand-written while-loop. The
behaviours are the same; what changed is where they live:
  _handle_tool_call()  → the @tool functions + LangGraph's ToolNode
  _agent_loop()        → the compiled graph (agent ⇄ tools)
  TOOLS (JSON schema)  → derived by LangChain from the @tool functions
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.errors import GraphRecursionError

from src.business.chatbot.agentic_chatbot import MAX_TOOL_ROUNDS, AgenticChatbot
from tests.conftest import FakeChatModel, FakeRedisMemory, reply, tool_call, tool_calls


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
class FakeLongTermMemory:
    """Records remember_conversation() calls; recall() returns canned hits."""

    def __init__(self, recall_results: List[Dict] | None = None):
        self._recall_results = recall_results or []
        self.recall_queries: List[str] = []
        self.rows: List[Dict] = []

    def recall(self, query: str, user_id: str, top_k: int = 5) -> List[Dict]:
        self.recall_queries.append(query)
        return list(self._recall_results)

    def remember_conversation(self, user_message: str, assistant_response: str, user_id: str) -> None:
        self.rows.append({
            "text": f"user: {user_message}\nassistant: {assistant_response}",
            "metadata": {"user_id": user_id, "type": "conversation"},
        })


def build_chatbot(
    scripted: List[AIMessage],
    *,
    recall_results: List[Dict] | None = None,
    live_results: List[Dict] | None = None,
    chat_history_manager=None,
    preload_turns: List[Dict] | None = None,
    user_info: Dict | None = None,
):
    """Construct an AgenticChatbot with every collaborator faked."""
    ltm = FakeLongTermMemory(recall_results)
    redis_memory = FakeRedisMemory(preload=preload_turns)

    live_provider = MagicMock()
    live_provider.search.return_value = live_results if live_results is not None else []

    llm = FakeChatModel(responses=list(scripted))
    bot = AgenticChatbot(
        long_term_memory=ltm,
        redis_memory=redis_memory,
        user_id="user-1",
        chat_history_manager=chat_history_manager,
        user_info=user_info,
        live_data_provider=live_provider,
        llm=llm,
    )
    return bot, ltm, redis_memory, live_provider


def tool_by_name(bot: AgenticChatbot, name: str):
    return next(t for t in bot.tools if t.name == name)


def run_graph(bot: AgenticChatbot, text: str = "?", **config):
    return bot.graph.invoke({"messages": [HumanMessage(content=text)]}, config=config or None)


# ---------------------------------------------------------------------------
# Provider guard
# ---------------------------------------------------------------------------
class TestProviderSelection:
    """LLM_PROVIDER must actually be honoured by the agent."""

    @pytest.fixture(autouse=True)
    def no_dotenv(self):
        # load_dotenv() would pull the developer's real .env into the test run
        # and could put back a credential a test deliberately removed.
        with patch("src.business.chatbot.agentic_chatbot.load_dotenv"), \
             patch("src.business.chatbot.agentic_chatbot.load_config", return_value={}):
            yield

    def _build(self):
        return AgenticChatbot(
            long_term_memory=MagicMock(),
            redis_memory=MagicMock(),
            user_id="u",
            live_data_provider=MagicMock(),
        )

    def test_huggingface_provider_raises_not_implemented(self, monkeypatch):
        """Local HF models can't do OpenAI-style tool calling — fail loud rather
        than answer without tools."""
        monkeypatch.setenv("LLM_PROVIDER", "huggingface")
        with pytest.raises(NotImplementedError, match="tool-calling graph"):
            self._build()

    def test_unknown_provider_raises_not_implemented(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "llama-cpp")
        with pytest.raises(NotImplementedError):
            self._build()

    def test_openai_provider_requires_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            self._build()

    def test_openai_provider_uses_the_configured_chat_model(self, monkeypatch, override_config):
        """Changing models.chat in config.yml is the whole change — no code edit."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        override_config({"models": {"chat": {"name": "gpt-4.1-mini"}}})
        assert self._build().model_name == "gpt-4.1-mini"

    def test_explicit_model_name_overrides_the_config(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        bot = AgenticChatbot(long_term_memory=MagicMock(), redis_memory=MagicMock(), user_id="u",
                             live_data_provider=MagicMock(), model_name="gpt-4.1")
        assert bot.model_name == "gpt-4.1"

    def test_azure_provider_requires_deployment_name(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
        monkeypatch.delenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", raising=False)
        with pytest.raises(RuntimeError, match="AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"):
            self._build()

    def test_azure_provider_builds_an_azure_chat_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
        monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "my-deployment")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        bot = self._build()
        assert type(bot.llm).__name__ == "AzureChatOpenAI"
        assert bot.model_name == "my-deployment"


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------
class TestToolSchema:
    """The tool definitions are the contract the model codes against.

    LangChain derives them from the @tool functions; these tests pin the
    schema the model actually receives (the OpenAI function format).
    """

    def test_exposes_all_tools(self):
        bot, *_ = build_chatbot([reply("ok")])
        assert {t.name for t in bot.tools} == {"search_vector_db", "web_search", "news_search"}

    def test_every_tool_declares_required_query_param(self):
        bot, *_ = build_chatbot([reply("ok")])
        for tool in bot.tools:
            fn = convert_to_openai_tool(tool)["function"]
            assert fn["parameters"]["required"] == ["query"]
            assert "query" in fn["parameters"]["properties"]
            assert fn["description"].strip(), f"{fn['name']} has no description"

    def test_tools_are_bound_to_the_model(self):
        bot, *_ = build_chatbot([reply("ok")])
        assert {t.name for t in bot.llm.bound_tools} == {"search_vector_db", "web_search", "news_search"}


# ---------------------------------------------------------------------------
# Tool behaviour (formerly _handle_tool_call)
# ---------------------------------------------------------------------------
class TestTools:
    def test_search_vector_db_returns_recalled_text(self):
        bot, *_ = build_chatbot(
            [reply("ok")],
            recall_results=[{"text": "user likes hiking", "metadata": {}, "score": 0.1},
                            {"text": "user lives in Toronto", "metadata": {}, "score": 0.2}],
        )
        out = tool_by_name(bot, "search_vector_db").invoke({"query": "hobbies"})
        assert "user likes hiking" in out
        assert "user lives in Toronto" in out

    def test_search_vector_db_with_no_hits_says_so(self):
        bot, *_ = build_chatbot([reply("ok")], recall_results=[])
        assert tool_by_name(bot, "search_vector_db").invoke({"query": "x"}) == \
            "No relevant past conversations found."

    def test_web_search_formats_results_with_source_and_url(self):
        bot, _, _, provider = build_chatbot(
            [reply("ok")],
            live_results=[{"title": "AI news", "summary": "Something happened",
                           "source": "Reuters", "url": "https://example.com/a"}],
        )
        out = tool_by_name(bot, "web_search").invoke({"query": "ai"})
        assert "AI news" in out
        assert "Reuters" in out
        assert "https://example.com/a" in out
        provider.search.assert_called_once_with(query="ai", limit=8)

    def test_news_search_passes_the_time_window_and_fetches_ten(self):
        """10 results, so "top 5" survives a few off-topic articles."""
        bot, _, _, provider = build_chatbot([reply("ok")])
        provider.news.return_value = [{"title": "AI story", "summary": "S", "source": "Reuters",
                                       "url": "https://r", "date": "2026-09-30"}]
        out = tool_by_name(bot, "news_search").invoke({"query": "AI", "days": 7})
        provider.news.assert_called_once_with(query="AI", days=7, limit=10)
        assert "AI story" in out and "2026-09-30" in out and "https://r" in out

    def test_news_search_defaults_to_the_last_week(self):
        bot, _, _, provider = build_chatbot([reply("ok")])
        provider.news.return_value = []
        tool_by_name(bot, "news_search").invoke({"query": "AI"})
        assert provider.news.call_args.kwargs["days"] == 7

    def test_news_search_error_is_contained(self):
        bot, _, _, provider = build_chatbot([reply("ok")])
        provider.news.side_effect = RuntimeError("quota exceeded")
        out = tool_by_name(bot, "news_search").invoke({"query": "AI"})
        assert "News search failed" in out and "quota exceeded" in out

    def test_web_search_omits_url_line_when_absent(self):
        bot, *_ = build_chatbot(
            [reply("ok")],
            live_results=[{"title": "T", "summary": "S", "source": "Src"}],
        )
        assert "URL:" not in tool_by_name(bot, "web_search").invoke({"query": "q"})

    def test_web_search_with_no_results_names_the_query(self):
        bot, *_ = build_chatbot([reply("ok")], live_results=[])
        assert "obscure thing" in tool_by_name(bot, "web_search").invoke({"query": "obscure thing"})

    def test_web_search_provider_exception_is_contained(self):
        """A failing search must degrade to a tool-result string, so the model
        can recover instead of the request 500-ing."""
        bot, _, _, provider = build_chatbot([reply("ok")])
        provider.search.side_effect = RuntimeError("network down")
        out = tool_by_name(bot, "web_search").invoke({"query": "q"})
        assert "Web search failed" in out
        assert "network down" in out

    def test_unknown_tool_name_is_reported_to_the_model_not_raised(self):
        """ToolNode answers a hallucinated tool name with an error ToolMessage."""
        bot, *_ = build_chatbot([tool_call("delete_everything", {}, "c1"), reply("sorry")])
        result = run_graph(bot)
        tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        assert "delete_everything" in tool_msg.content
        assert tool_msg.status == "error"
        assert result["messages"][-1].content == "sorry"


# ---------------------------------------------------------------------------
# The agent graph (formerly _agent_loop)
# ---------------------------------------------------------------------------
class TestAgentGraph:
    """agent → (tools → agent)* → END"""

    def test_graph_has_agent_and_tools_nodes(self):
        bot, *_ = build_chatbot([reply("ok")])
        assert {"agent", "tools"} <= set(bot.graph.get_graph().nodes)

    def test_returns_immediately_when_no_tool_calls(self):
        bot, *_ = build_chatbot([reply("direct answer")])
        assert run_graph(bot)["messages"][-1].content == "direct answer"
        assert len(bot.llm.calls) == 1

    def test_runs_one_tool_then_answers(self):
        bot, *_ = build_chatbot(
            [tool_call("web_search", {"query": "news"}), reply("final answer")],
            live_results=[{"title": "T", "summary": "S", "source": "Src", "url": ""}],
        )
        assert run_graph(bot)["messages"][-1].content == "final answer"
        assert len(bot.llm.calls) == 2

    def test_tool_result_is_fed_back_to_the_model(self):
        """The whole point of the loop: call 2 must SEE call 1's tool output."""
        bot, *_ = build_chatbot(
            [tool_call("web_search", {"query": "news"}, call_id="abc"), reply("done")],
            live_results=[{"title": "HEADLINE", "summary": "S", "source": "Src", "url": ""}],
        )
        run_graph(bot)

        tool_messages = [m for m in bot.llm.calls[1] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "abc"
        assert "HEADLINE" in tool_messages[0].content

    def test_handles_parallel_tool_calls_in_one_message(self):
        """One AI message can request several tools at once — each gets its own
        ToolMessage keyed by tool_call_id."""
        bot, *_ = build_chatbot(
            [tool_calls([("search_vector_db", {"query": "past"}, "c1"),
                         ("web_search", {"query": "now"}, "c2")]),
             reply("combined")],
            recall_results=[{"text": "remembered", "metadata": {}, "score": 0.1}],
            live_results=[{"title": "fresh", "summary": "s", "source": "src", "url": ""}],
        )
        assert run_graph(bot)["messages"][-1].content == "combined"

        tool_ids = [m.tool_call_id for m in bot.llm.calls[1] if isinstance(m, ToolMessage)]
        assert tool_ids == ["c1", "c2"]

    def test_multi_hop_tool_use(self):
        """The graph must survive more than one round-trip."""
        bot, *_ = build_chatbot(
            [tool_call("search_vector_db", {"query": "a"}, "c1"),
             tool_call("web_search", {"query": "b"}, "c2"),
             reply("after two hops")],
            recall_results=[{"text": "x", "metadata": {}, "score": 0.1}],
            live_results=[{"title": "y", "summary": "s", "source": "src", "url": ""}],
        )
        assert run_graph(bot)["messages"][-1].content == "after two hops"
        assert len(bot.llm.calls) == 3

    def test_max_tool_rounds_comes_from_the_config(self):
        from src.utils.config import load_config

        assert MAX_TOOL_ROUNDS == load_config()["agent"]["max_tool_rounds"]

    def test_runaway_tool_loop_is_stopped_after_max_tool_rounds(self):
        """cortex-core's while-loop had no cap: a model that kept requesting
        tools looped until the process was killed. The graph is compiled with
        recursion_limit sized for MAX_TOOL_ROUNDS, so it stops and raises."""
        scripted = [tool_call("web_search", {"query": f"q{i}"}, f"c{i}") for i in range(50)]
        bot, *_ = build_chatbot(scripted)

        with pytest.raises(GraphRecursionError):
            run_graph(bot)
        assert len(bot.llm.calls) == MAX_TOOL_ROUNDS + 1

    def test_max_tool_rounds_is_allowed(self):
        """The cap is an upper bound, not an off-by-one: exactly MAX_TOOL_ROUNDS
        tool rounds followed by an answer still succeeds."""
        scripted = [tool_call("web_search", {"query": f"q{i}"}, f"c{i}") for i in range(MAX_TOOL_ROUNDS)]
        bot, *_ = build_chatbot(scripted + [reply("made it")])
        assert run_graph(bot)["messages"][-1].content == "made it"

    def test_recursion_limit_is_configurable_per_run(self):
        scripted = [tool_call("web_search", {"query": f"q{i}"}, f"c{i}") for i in range(10)]
        bot, *_ = build_chatbot(scripted)
        with pytest.raises(GraphRecursionError):
            run_graph(bot, recursion_limit=4)
        assert len(bot.llm.calls) == 2


# ---------------------------------------------------------------------------
# chat() — context assembly
# ---------------------------------------------------------------------------
class TestChatContextAssembly:
    @pytest.mark.asyncio
    async def test_system_prompt_is_first_and_user_message_last(self):
        bot, *_ = build_chatbot([reply("hi")])
        await bot.chat("hello there")

        msgs = bot.llm.calls[0]
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[-1], HumanMessage)
        assert msgs[-1].content == "hello there"

    @pytest.mark.asyncio
    async def test_recent_redis_turns_are_replayed_between_system_and_user(self):
        bot, *_ = build_chatbot(
            [reply("hi")],
            preload_turns=[{"role": "user", "content": "earlier q"},
                           {"role": "assistant", "content": "earlier a"}],
        )
        await bot.chat("new question")

        msgs = bot.llm.calls[0]
        assert [type(m) for m in msgs[1:3]] == [HumanMessage, AIMessage]
        assert [m.content for m in msgs[1:3]] == ["earlier q", "earlier a"]

    @pytest.mark.asyncio
    async def test_prefetched_memories_land_in_the_system_prompt(self):
        """The proactive recall must reach the model, not just be computed."""
        bot, ltm, _, _ = build_chatbot(
            [reply("hi")],
            recall_results=[{"text": "user is a data engineer", "metadata": {}, "score": 0.1}],
        )
        await bot.chat("what should I learn next?")

        assert ltm.recall_queries == ["what should I learn next?"]
        assert "user is a data engineer" in bot.llm.calls[0][0].content

    @pytest.mark.asyncio
    async def test_user_info_lands_in_the_system_prompt(self):
        bot, *_ = build_chatbot([reply("hi")], user_info={"name": "Amir", "city": "Toronto"})
        await bot.chat("hey")
        system_prompt = bot.llm.calls[0][0].content
        assert "Amir" in system_prompt and "Toronto" in system_prompt

    @pytest.mark.asyncio
    async def test_cold_session_hydrates_summary_from_sqlite(self):
        """Fresh Redis + a history manager → pull the previous session forward."""
        manager = MagicMock()
        manager.get_latest_summary.return_value = "User: prior\nAssistant: context"
        bot, *_ = build_chatbot([reply("hi")], chat_history_manager=manager)
        await bot.chat("continue")

        manager.get_latest_summary.assert_called_once_with("user-1")
        assert "prior" in bot.llm.calls[0][0].content

    @pytest.mark.asyncio
    async def test_warm_session_does_not_hydrate_summary(self):
        """Redis already has the context — re-reading SQLite would duplicate it."""
        manager = MagicMock()
        bot, *_ = build_chatbot(
            [reply("hi")], chat_history_manager=manager,
            preload_turns=[{"role": "user", "content": "existing"}],
        )
        await bot.chat("more")
        manager.get_latest_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_summary_suppresses_sqlite_lookup(self):
        manager = MagicMock()
        bot, *_ = build_chatbot([reply("hi")], chat_history_manager=manager)
        await bot.chat("q", chat_summary="caller-supplied summary")
        manager.get_latest_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_id_defaults_to_user_id(self):
        bot, _, redis_memory, _ = build_chatbot([reply("hi")])
        await bot.chat("q")
        assert "user-1" in redis_memory.store


# ---------------------------------------------------------------------------
# chat() — the 3-way persistence fan-out
# ---------------------------------------------------------------------------
class TestChatPersistence:
    """Every turn must land in Redis, the vector store, AND SQLite."""

    @pytest.mark.asyncio
    async def test_persists_both_roles_to_redis(self):
        bot, _, redis_memory, _ = build_chatbot([reply("the answer")])
        await bot.chat("the question", session_id="s1")

        assert redis_memory.store["s1"] == [
            {"role": "user", "content": "the question"},
            {"role": "assistant", "content": "the answer"},
        ]

    @pytest.mark.asyncio
    async def test_persists_exchange_to_long_term_memory(self):
        bot, ltm, _, _ = build_chatbot([reply("the answer")])
        await bot.chat("the question")

        assert len(ltm.rows) == 1
        row = ltm.rows[0]
        assert "the question" in row["text"] and "the answer" in row["text"]
        assert row["metadata"]["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_persists_to_sqlite_with_session_ensured_first(self):
        manager = MagicMock()
        manager.get_latest_summary.return_value = None
        bot, *_ = build_chatbot([reply("A")], chat_history_manager=manager)
        await bot.chat("Q", session_id="s9")

        manager.ensure_session.assert_called_once()
        assert manager.save_message.call_count == 2
        roles = [c.args[2] for c in manager.save_message.call_args_list]
        assert roles == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_first_turn_titles_the_session_from_the_message(self):
        manager = MagicMock()
        manager.get_latest_summary.return_value = None
        bot, *_ = build_chatbot([reply("A")], chat_history_manager=manager)
        await bot.chat("How do I deploy this to Azure?", session_id="s1")

        assert manager.ensure_session.call_args.args[2] == "How do I deploy this to Azure?"

    @pytest.mark.asyncio
    async def test_long_first_message_is_truncated_to_60_chars(self):
        manager = MagicMock()
        manager.get_latest_summary.return_value = None
        bot, *_ = build_chatbot([reply("A")], chat_history_manager=manager)
        await bot.chat("x" * 200, session_id="s1")

        assert len(manager.ensure_session.call_args.args[2]) == 60

    @pytest.mark.asyncio
    async def test_later_turns_do_not_retitle_the_session(self):
        manager = MagicMock()
        bot, *_ = build_chatbot(
            [reply("A")], chat_history_manager=manager,
            preload_turns=[{"role": "user", "content": "earlier"}],
        )
        await bot.chat("second question", session_id="s1")

        assert manager.ensure_session.call_args.args[2] == "New conversation"

    @pytest.mark.asyncio
    async def test_sqlite_is_optional(self):
        """No history manager → Redis + vector store still work, no crash."""
        bot, ltm, redis_memory, _ = build_chatbot([reply("A")], chat_history_manager=None)
        assert await bot.chat("Q") == "A"
        assert len(ltm.rows) == 1
        assert len(redis_memory.store["user-1"]) == 2

    @pytest.mark.asyncio
    async def test_tool_using_turn_persists_the_final_answer_only(self):
        """Intermediate tool chatter must not pollute long-term memory."""
        bot, ltm, redis_memory, _ = build_chatbot(
            [tool_call("web_search", {"query": "news"}), reply("summarised answer")],
            live_results=[{"title": "T", "summary": "S", "source": "Src", "url": ""}],
        )
        await bot.chat("what's new?", session_id="s1")

        assert len(ltm.rows) == 1
        assert ltm.rows[0]["text"] == "user: what's new?\nassistant: summarised answer"
        assert redis_memory.store["s1"][1]["content"] == "summarised answer"

    @pytest.mark.asyncio
    async def test_returns_the_models_final_content(self):
        bot, *_ = build_chatbot([reply("returned to caller")])
        assert await bot.chat("q") == "returned to caller"


# ---------------------------------------------------------------------------
# chat() — token usage (new: cortex-core never reported any)
# ---------------------------------------------------------------------------
class TestTokenUsage:
    @pytest.mark.asyncio
    async def test_usage_is_summed_across_every_model_call_in_the_turn(self):
        bot, *_ = build_chatbot([
            tool_call("web_search", {"query": "q"}, usage={"input_tokens": 100, "output_tokens": 10}),
            reply("done", usage={"input_tokens": 150, "output_tokens": 30}),
        ])
        await bot.chat("q")
        assert bot.last_usage == {"input_tokens": 250, "output_tokens": 40, "total_tokens": 290}

    @pytest.mark.asyncio
    async def test_missing_usage_counts_as_zero(self):
        bot, *_ = build_chatbot([reply("done")])
        await bot.chat("q")
        assert bot.last_usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def test_model_name_comes_from_the_model(self):
        bot, *_ = build_chatbot([reply("ok")])
        assert bot.model_name == "fake-model"
