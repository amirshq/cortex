"""The agent's tools — LangChain `@tool` edition.

In cortex-core each tool existed twice: once as a hand-written JSON schema in
`AgenticChatbot.TOOLS`, and once as a branch of `_handle_tool_call()`'s
`if tool_name == ...` dispatch. Keeping the two in sync was manual.

With `@tool`, the function IS the tool: LangChain derives the JSON schema from
the signature + docstring, and LangGraph's `ToolNode` does the dispatch. The
tool names and descriptions below are unchanged from cortex-core, so the model
sees exactly the same tool definitions.

The tools need per-request context (which user, which memory store, which
search provider). They get it by closure: `build_tools()` is called once per
chatbot, and the inner functions capture those objects.
"""

from __future__ import annotations

from typing import List

from langchain_core.tools import BaseTool, tool

from src.business.core.live_data import LiveDataProvider
from src.memory.long_term_memory import LongTermMemory


def build_tools(
    long_term_memory: LongTermMemory,
    live_data_provider: LiveDataProvider,
    user_id: str,
) -> List[BaseTool]:
    """Create the agent's tools, bound to one user's memory and a search provider."""

    @tool
    def search_vector_db(query: str) -> str:
        """Semantic search over all past conversations. Use this when you need to recall what the user said, asked, or was told in a previous session.

        Args:
            query: Natural-language description of what to look for.
        """
        results = long_term_memory.recall(query=query, user_id=user_id)
        if not results:
            return "No relevant past conversations found."
        return "\n\n".join(r["text"] for r in results)

    def format_results(results, query: str) -> str:
        """Numbered, readable results: title, source, date, summary, URL."""
        if not results:
            return f"No search results found for '{query}'."
        formatted_results = []
        for i, result in enumerate(results, 1):
            source = result.get("source", "Unknown")
            if result.get("date"):
                source += f" · {result['date']}"
            text = f"{i}. {result.get('title', '')}\n   Source: {source}\n   {result.get('summary', '')}"
            if result.get("url"):
                text += f"\n   URL: {result['url']}"
            formatted_results.append(text)
        return "\n\n".join(formatted_results)

    @tool
    def web_search(query: str) -> str:
        """Search the general web: explanations, documentation, definitions, how-tos, comparisons, and facts that may have changed since your training (e.g. 'what is LangGraph', 'LangChain vs LangGraph', 'Python 3.13 release date'). For recent news, use news_search instead.

        Args:
            query: What to search for, in the user's own words.
        """
        try:
            # 8 results: enough to cover the topic even when a few are off-target.
            return format_results(live_data_provider.search(query=query, limit=8), query)
        except Exception as e:
            return f"Web search failed: {str(e)}"

    @tool
    def news_search(query: str, days: int = 7) -> str:
        """Find recent NEWS articles published in the last `days` days. Use this for current events, "latest", "this week", "today", or trending topics (e.g. 'top AI news this week' → query='artificial intelligence', days=7).

        Args:
            query: The news TOPIC, spelled out in full ('artificial intelligence', not 'AI'). Leave out ranking words like 'top', 'hot', 'latest', 'trending' and any dates — they match unrelated articles.
            days: How many days back to search (1 = today, 7 = this week, 30 = this month).
        """
        try:
            # 10 results, so a request for "top 5" still has 5 relevant ones
            # after the off-topic articles are discarded.
            return format_results(live_data_provider.news(query=query, days=days, limit=10), query)
        except Exception as e:
            return f"News search failed: {str(e)}"

    return [search_vector_db, web_search, news_search]
