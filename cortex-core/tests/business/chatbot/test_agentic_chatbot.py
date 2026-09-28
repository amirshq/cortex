"""Tests for AgenticChatbot — the tool-calling loop and the memory fan-out.

This is the orchestration layer: the code that decides which tools run,
how many times the model is called, and what gets persisted where. None
of it was covered before, and none of it is exercised by the controller
tests (which mock `process_chat_message` wholesale).

Everything here is hermetic — the OpenAI client, Redis, Chroma and SQLite
are all replaced with the in-memory fakes from conftest.py.
"""

from __future__ import annotations

import json
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.business.chatbot.agentic_chatbot import AgenticChatbot
from src.memory.long_term_memory import LongTermMemory
from tests.conftest import (
    FakeConversationVectorStore,
    FakeEmbedder,
    FakeMessage,
    FakeOpenAIClient,
    FakeRedisMemory,
    FakeToolCall,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_chatbot(
    monkeypatch,
    scripted: List[FakeMessage],
    *,
    recall_results: List[Dict] | None = None,
    live_results: List[Dict] | None = None,
    chat_history_manager=None,
    preload_turns: List[Dict] | None = None,
    user_info: Dict | None = None,
):
    """Construct an AgenticChatbot with every collaborator faked."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    store = FakeConversationVectorStore(search_results=recall_results)
    embedder = FakeEmbedder()

    class _IdentityChunker:
        def split(self, text): return [text]

    ltm = LongTermMemory(vectordb=store, embedder=embedder, chunker=_IdentityChunker())
    redis_memory = FakeRedisMemory(preload=preload_turns)

    live_provider = MagicMock()
    live_provider.search.return_value = live_results if live_results is not None else []

    # load_dotenv() would pull the developer's real .env into the test run
    # and could flip LLM_PROVIDER out from under monkeypatch.
    with patch("src.business.chatbot.agentic_chatbot.load_dotenv"), \
         patch("src.business.chatbot.agentic_chatbot.OpenAI"), \
         patch("src.business.chatbot.agentic_chatbot.load_config", return_value={}):
        bot = AgenticChatbot(
            long_term_memory=ltm,
            redis_memory=redis_memory,
            user_id="user-1",
            chat_history_manager=chat_history_manager,
            user_info=user_info,
            live_data_provider=live_provider,
        )

    bot.client = FakeOpenAIClient(scripted)
    return bot, store, redis_memory, live_provider


def text_reply(content: str) -> FakeMessage:
    return FakeMessage(content=content)


def tool_reply(name: str, args: Dict, call_id: str = "call-1") -> FakeMessage:
    return FakeMessage(tool_calls=[FakeToolCall(call_id, name, json.dumps(args))])


# ---------------------------------------------------------------------------
# Provider guard
# ---------------------------------------------------------------------------
class TestProviderSelection:
    """LLM_PROVIDER must actually be honoured by the agent loop."""

    def test_huggingface_provider_raises_not_implemented(self, monkeypatch):
        """HF models can't do OpenAI-style function calling — fail loud.

        This used to silently construct an OpenAI client regardless of the
        setting, which meant an on-prem deployment believing it ran locally
        was actually calling out to OpenAI.
        """
        monkeypatch.setenv("LLM_PROVIDER", "huggingface")
        with patch("src.business.chatbot.agentic_chatbot.load_dotenv"), \
             patch("src.business.chatbot.agentic_chatbot.load_config", return_value={}):
            with pytest.raises(NotImplementedError, match="tool-calling loop"):
                AgenticChatbot(
                    long_term_memory=MagicMock(),
                    redis_memory=MagicMock(),
                    user_id="u",
                    live_data_provider=MagicMock(),
                )

    def test_unknown_provider_raises_not_implemented(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "llama-cpp")
        with patch("src.business.chatbot.agentic_chatbot.load_dotenv"), \
             patch("src.business.chatbot.agentic_chatbot.load_config", return_value={}):
            with pytest.raises(NotImplementedError):
                AgenticChatbot(
                    long_term_memory=MagicMock(),
                    redis_memory=MagicMock(),
                    user_id="u",
                    live_data_provider=MagicMock(),
                )

    def test_openai_provider_requires_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("src.business.chatbot.agentic_chatbot.load_dotenv"), \
             patch("src.business.chatbot.agentic_chatbot.load_config", return_value={}):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                AgenticChatbot(
                    long_term_memory=MagicMock(),
                    redis_memory=MagicMock(),
                    user_id="u",
                    live_data_provider=MagicMock(),
                )

    def test_azure_provider_requires_deployment_name(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
        monkeypatch.delenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", raising=False)
        with patch("src.business.chatbot.agentic_chatbot.load_dotenv"), \
             patch("src.business.chatbot.agentic_chatbot.load_config", return_value={}):
            with pytest.raises(RuntimeError, match="AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"):
                AgenticChatbot(
                    long_term_memory=MagicMock(),
                    redis_memory=MagicMock(),
                    user_id="u",
                    live_data_provider=MagicMock(),
                )

    def test_azure_provider_uses_shared_client_builder(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
        monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "my-deployment")
        with patch("src.business.chatbot.agentic_chatbot.load_dotenv"), \
             patch("src.business.chatbot.agentic_chatbot.load_config", return_value={}), \
             patch("src.business.chatbot.agentic_chatbot.build_azure_openai_client") as mock_build:
            bot = AgenticChatbot(
                long_term_memory=MagicMock(),
                redis_memory=MagicMock(),
                user_id="u",
                live_data_provider=MagicMock(),
            )
        mock_build.assert_called_once()
        assert bot.model_name == "my-deployment"


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------
class TestToolSchema:
    """The tool definitions are the contract the model codes against."""

    def test_exposes_both_tools(self):
        names = {t["function"]["name"] for t in AgenticChatbot.TOOLS}
        assert names == {"search_vector_db", "web_search"}

    def test_every_tool_declares_required_query_param(self):
        for tool in AgenticChatbot.TOOLS:
            fn = tool["function"]
            assert tool["type"] == "function"
            assert fn["parameters"]["required"] == ["query"]
            assert "query" in fn["parameters"]["properties"]
            assert fn["description"].strip(), f"{fn['name']} has no description"


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
class TestToolDispatch:
    """_handle_tool_call routes names to implementations and formats results."""

    def test_search_vector_db_returns_recalled_text(self, monkeypatch):
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("ok")],
            recall_results=[{"text": "user likes hiking", "metadata": {}, "score": 0.1},
                            {"text": "user lives in Toronto", "metadata": {}, "score": 0.2}],
        )
        out = bot._handle_tool_call("search_vector_db", {"query": "hobbies"})
        assert "user likes hiking" in out
        assert "user lives in Toronto" in out

    def test_search_vector_db_with_no_hits_says_so(self, monkeypatch):
        bot, _, _, _ = build_chatbot(monkeypatch, [text_reply("ok")], recall_results=[])
        assert bot._handle_tool_call("search_vector_db", {"query": "x"}) == \
            "No relevant past conversations found."

    def test_web_search_formats_results_with_source_and_url(self, monkeypatch):
        bot, _, _, provider = build_chatbot(
            monkeypatch, [text_reply("ok")],
            live_results=[{"title": "AI news", "summary": "Something happened",
                           "source": "Reuters", "url": "https://example.com/a"}],
        )
        out = bot._handle_tool_call("web_search", {"query": "ai"})
        assert "AI news" in out
        assert "Reuters" in out
        assert "https://example.com/a" in out
        provider.search.assert_called_once_with(query="ai", limit=5)

    def test_web_search_omits_url_line_when_absent(self, monkeypatch):
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("ok")],
            live_results=[{"title": "T", "summary": "S", "source": "Src"}],
        )
        assert "URL:" not in bot._handle_tool_call("web_search", {"query": "q"})

    def test_web_search_with_no_results_names_the_query(self, monkeypatch):
        bot, _, _, _ = build_chatbot(monkeypatch, [text_reply("ok")], live_results=[])
        out = bot._handle_tool_call("web_search", {"query": "obscure thing"})
        assert "obscure thing" in out

    def test_web_search_provider_exception_is_contained(self, monkeypatch):
        """A failing search must degrade to a tool-result string.

        If this raised, it would escape _agent_loop and 500 the request
        instead of letting the model recover or say it couldn't look it up.
        """
        bot, _, _, provider = build_chatbot(monkeypatch, [text_reply("ok")])
        provider.search.side_effect = RuntimeError("network down")
        out = bot._handle_tool_call("web_search", {"query": "q"})
        assert "Web search failed" in out
        assert "network down" in out

    def test_unknown_tool_name_returns_marker_not_raise(self, monkeypatch):
        bot, _, _, _ = build_chatbot(monkeypatch, [text_reply("ok")])
        assert bot._handle_tool_call("delete_everything", {}) == "Unknown tool: delete_everything"


# ---------------------------------------------------------------------------
# The ReAct loop
# ---------------------------------------------------------------------------
class TestAgentLoop:
    """_agent_loop: call model → run tools → feed results back → repeat."""

    def test_returns_immediately_when_no_tool_calls(self, monkeypatch):
        bot, _, _, _ = build_chatbot(monkeypatch, [text_reply("direct answer")])
        assert bot._agent_loop([{"role": "user", "content": "hi"}]) == "direct answer"
        assert len(bot.client.completions.calls) == 1

    def test_runs_one_tool_then_answers(self, monkeypatch):
        bot, _, _, _ = build_chatbot(
            monkeypatch,
            [tool_reply("web_search", {"query": "news"}), text_reply("final answer")],
            live_results=[{"title": "T", "summary": "S", "source": "Src", "url": ""}],
        )
        assert bot._agent_loop([{"role": "user", "content": "news?"}]) == "final answer"
        assert len(bot.client.completions.calls) == 2

    def test_tool_result_is_fed_back_to_the_model(self, monkeypatch):
        """The whole point of the loop: turn 2 must SEE turn 1's tool output."""
        bot, _, _, _ = build_chatbot(
            monkeypatch,
            [tool_reply("web_search", {"query": "news"}, call_id="abc"), text_reply("done")],
            live_results=[{"title": "HEADLINE", "summary": "S", "source": "Src", "url": ""}],
        )
        bot._agent_loop([{"role": "user", "content": "news?"}])

        second_call_messages = bot.client.completions.calls[1]["messages"]
        tool_messages = [m for m in second_call_messages
                         if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "abc"
        assert "HEADLINE" in tool_messages[0]["content"]

    def test_handles_parallel_tool_calls_in_one_message(self, monkeypatch):
        """One assistant message can request several tools at once — each
        needs its own tool-result message keyed by tool_call_id."""
        both = FakeMessage(tool_calls=[
            FakeToolCall("c1", "search_vector_db", json.dumps({"query": "past"})),
            FakeToolCall("c2", "web_search", json.dumps({"query": "now"})),
        ])
        bot, _, _, _ = build_chatbot(
            monkeypatch, [both, text_reply("combined")],
            recall_results=[{"text": "remembered", "metadata": {}, "score": 0.1}],
            live_results=[{"title": "fresh", "summary": "s", "source": "src", "url": ""}],
        )
        assert bot._agent_loop([{"role": "user", "content": "?"}]) == "combined"

        msgs = bot.client.completions.calls[1]["messages"]
        tool_ids = [m["tool_call_id"] for m in msgs
                    if isinstance(m, dict) and m.get("role") == "tool"]
        assert tool_ids == ["c1", "c2"]

    def test_multi_hop_tool_use(self, monkeypatch):
        """The loop must survive more than one round-trip."""
        bot, _, _, _ = build_chatbot(
            monkeypatch,
            [tool_reply("search_vector_db", {"query": "a"}, "c1"),
             tool_reply("web_search", {"query": "b"}, "c2"),
             text_reply("after two hops")],
            recall_results=[{"text": "x", "metadata": {}, "score": 0.1}],
            live_results=[{"title": "y", "summary": "s", "source": "src", "url": ""}],
        )
        assert bot._agent_loop([{"role": "user", "content": "?"}]) == "after two hops"
        assert len(bot.client.completions.calls) == 3

    def test_tools_and_auto_choice_are_sent_every_call(self, monkeypatch):
        bot, _, _, _ = build_chatbot(
            monkeypatch,
            [tool_reply("web_search", {"query": "q"}), text_reply("done")],
        )
        bot._agent_loop([{"role": "user", "content": "?"}])
        for call in bot.client.completions.calls:
            assert call["tool_choice"] == "auto"
            assert call["tools"] == AgenticChatbot.TOOLS

    def test_loop_has_no_iteration_cap(self, monkeypatch):
        """DOCUMENTS A KNOWN GAP, it does not endorse it.

        `_agent_loop` is a `while True:` with no max-iteration guard. A model
        that keeps requesting tools loops until the process is killed. This
        test pins the current behaviour so that adding a cap (or porting to
        LangGraph's recursion_limit) is a deliberate, visible change: it will
        fail and must be rewritten to assert the new limit.
        """
        scripted = [tool_reply("web_search", {"query": f"q{i}"}, f"c{i}") for i in range(50)]
        scripted.append(text_reply("finally"))
        bot, _, _, _ = build_chatbot(monkeypatch, scripted)

        assert bot._agent_loop([{"role": "user", "content": "?"}]) == "finally"
        assert len(bot.client.completions.calls) == 51


# ---------------------------------------------------------------------------
# chat() — context assembly
# ---------------------------------------------------------------------------
class TestChatContextAssembly:
    @pytest.mark.asyncio
    async def test_system_prompt_is_first_and_user_message_last(self, monkeypatch):
        bot, _, _, _ = build_chatbot(monkeypatch, [text_reply("hi")])
        await bot.chat("hello there")

        msgs = bot.client.completions.calls[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[-1] == {"role": "user", "content": "hello there"}

    @pytest.mark.asyncio
    async def test_recent_redis_turns_are_replayed_between_system_and_user(self, monkeypatch):
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("hi")],
            preload_turns=[{"role": "user", "content": "earlier q"},
                           {"role": "assistant", "content": "earlier a"}],
        )
        await bot.chat("new question")

        msgs = bot.client.completions.calls[0]["messages"]
        assert [m["content"] for m in msgs[1:3]] == ["earlier q", "earlier a"]

    @pytest.mark.asyncio
    async def test_prefetched_memories_land_in_the_system_prompt(self, monkeypatch):
        """The proactive recall must reach the model, not just be computed."""
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("hi")],
            recall_results=[{"text": "user is a data engineer", "metadata": {}, "score": 0.1}],
        )
        await bot.chat("what should I learn next?")

        system_prompt = bot.client.completions.calls[0]["messages"][0]["content"]
        assert "user is a data engineer" in system_prompt

    @pytest.mark.asyncio
    async def test_user_info_lands_in_the_system_prompt(self, monkeypatch):
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("hi")], user_info={"name": "Amir", "city": "Toronto"},
        )
        await bot.chat("hey")
        system_prompt = bot.client.completions.calls[0]["messages"][0]["content"]
        assert "Amir" in system_prompt and "Toronto" in system_prompt

    @pytest.mark.asyncio
    async def test_cold_session_hydrates_summary_from_sqlite(self, monkeypatch):
        """Fresh Redis + a history manager → pull the previous session forward."""
        manager = MagicMock()
        manager.get_latest_summary.return_value = "User: prior\nAssistant: context"
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("hi")], chat_history_manager=manager,
        )
        await bot.chat("continue")

        manager.get_latest_summary.assert_called_once_with("user-1")
        assert "prior" in bot.client.completions.calls[0]["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_warm_session_does_not_hydrate_summary(self, monkeypatch):
        """Redis already has the context — re-reading SQLite would duplicate it."""
        manager = MagicMock()
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("hi")], chat_history_manager=manager,
            preload_turns=[{"role": "user", "content": "existing"}],
        )
        await bot.chat("more")
        manager.get_latest_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_summary_suppresses_sqlite_lookup(self, monkeypatch):
        manager = MagicMock()
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("hi")], chat_history_manager=manager,
        )
        await bot.chat("q", chat_summary="caller-supplied summary")
        manager.get_latest_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_id_defaults_to_user_id(self, monkeypatch):
        bot, _, redis_memory, _ = build_chatbot(monkeypatch, [text_reply("hi")])
        await bot.chat("q")
        assert "user-1" in redis_memory.store


# ---------------------------------------------------------------------------
# chat() — the 3-way persistence fan-out
# ---------------------------------------------------------------------------
class TestChatPersistence:
    """Every turn must land in Redis, the vector store, AND SQLite."""

    @pytest.mark.asyncio
    async def test_persists_both_roles_to_redis(self, monkeypatch):
        bot, _, redis_memory, _ = build_chatbot(monkeypatch, [text_reply("the answer")])
        await bot.chat("the question", session_id="s1")

        assert redis_memory.store["s1"] == [
            {"role": "user", "content": "the question"},
            {"role": "assistant", "content": "the answer"},
        ]

    @pytest.mark.asyncio
    async def test_persists_exchange_to_vector_store(self, monkeypatch):
        bot, store, _, _ = build_chatbot(monkeypatch, [text_reply("the answer")])
        await bot.chat("the question")

        assert len(store.rows) == 1
        row = store.rows[0]
        assert "the question" in row["text"] and "the answer" in row["text"]
        assert row["metadata"]["user_id"] == "user-1"
        assert row["metadata"]["type"] == "conversation"

    @pytest.mark.asyncio
    async def test_persists_to_sqlite_with_session_ensured_first(self, monkeypatch):
        manager = MagicMock()
        manager.get_latest_summary.return_value = None
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("A")], chat_history_manager=manager,
        )
        await bot.chat("Q", session_id="s9")

        manager.ensure_session.assert_called_once()
        assert manager.save_message.call_count == 2
        roles = [c.args[2] for c in manager.save_message.call_args_list]
        assert roles == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_first_turn_titles_the_session_from_the_message(self, monkeypatch):
        manager = MagicMock()
        manager.get_latest_summary.return_value = None
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("A")], chat_history_manager=manager,
        )
        await bot.chat("How do I deploy this to Azure?", session_id="s1")

        assert manager.ensure_session.call_args.args[2] == "How do I deploy this to Azure?"

    @pytest.mark.asyncio
    async def test_long_first_message_is_truncated_to_60_chars(self, monkeypatch):
        manager = MagicMock()
        manager.get_latest_summary.return_value = None
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("A")], chat_history_manager=manager,
        )
        await bot.chat("x" * 200, session_id="s1")

        assert len(manager.ensure_session.call_args.args[2]) == 60

    @pytest.mark.asyncio
    async def test_later_turns_do_not_retitle_the_session(self, monkeypatch):
        manager = MagicMock()
        bot, _, _, _ = build_chatbot(
            monkeypatch, [text_reply("A")], chat_history_manager=manager,
            preload_turns=[{"role": "user", "content": "earlier"}],
        )
        await bot.chat("second question", session_id="s1")

        assert manager.ensure_session.call_args.args[2] == "New conversation"

    @pytest.mark.asyncio
    async def test_sqlite_is_optional(self, monkeypatch):
        """No history manager → Redis + vector store still work, no crash."""
        bot, store, redis_memory, _ = build_chatbot(
            monkeypatch, [text_reply("A")], chat_history_manager=None,
        )
        assert await bot.chat("Q") == "A"
        assert len(store.rows) == 1
        assert len(redis_memory.store["user-1"]) == 2

    @pytest.mark.asyncio
    async def test_tool_using_turn_persists_the_final_answer_only(self, monkeypatch):
        """Intermediate tool chatter must not pollute long-term memory."""
        bot, store, redis_memory, _ = build_chatbot(
            monkeypatch,
            [tool_reply("web_search", {"query": "news"}), text_reply("summarised answer")],
            live_results=[{"title": "T", "summary": "S", "source": "Src", "url": ""}],
        )
        await bot.chat("what's new?", session_id="s1")

        assert len(store.rows) == 1
        assert store.rows[0]["text"] == "user: what's new?\nassistant: summarised answer"
        assert redis_memory.store["s1"][1]["content"] == "summarised answer"

    @pytest.mark.asyncio
    async def test_returns_the_models_final_content(self, monkeypatch):
        bot, _, _, _ = build_chatbot(monkeypatch, [text_reply("returned to caller")])
        assert await bot.chat("q") == "returned to caller"
