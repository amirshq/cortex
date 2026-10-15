"""Live data providers for real-time information.

Every provider answers two kinds of question, behind two agent tools:

  search(query)       — the general web: explanations, docs, facts
                        ("what is LangGraph?")        → the web_search tool
  news(query, days)   — recent news within a time window
                        ("AI hot topics this week")    → the news_search tool

They are different jobs. A news API can't explain what LangGraph is (it only
finds articles that mention it), and a general web search is poor at "only the
last 7 days". So each provider does what it is good at, and delegates the rest.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Optional, List, Dict, Any
import json

try:
    import requests
except ImportError:
    requests = None

try:
    # `ddgs` is the maintained successor of the `duckduckgo-search` package,
    # whose old DuckDuckGo endpoints stopped working. Its default "auto"
    # backend falls back across search engines when one blocks the request.
    from ddgs import DDGS
except ImportError:
    DDGS = None


class LiveDataProvider(ABC):
    """Abstract base class for live data providers."""

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for live data matching the query.

        Args:
            query: Search query (e.g., "latest COVID-19 cases", "weather in NYC")
            limit: Maximum number of results to return

        Returns:
            List of result dicts, each with at least:
            - "title": str
            - "summary": str
            - "source": str (optional)
            - "url": str (optional)
        """
        pass

    def news(self, query: str, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Recent news about `query` from the last `days` days.

        Default: fall back to a general search. Providers with a real news
        source override this. Results may also carry "date" (YYYY-MM-DD).
        """
        return self.search(query, limit=limit)


class DuckDuckGoSearchProvider(LiveDataProvider):
    """
    Web and news search via the `ddgs` library (free, no API key required).
    """

    def __init__(self):
        if DDGS is None:
            raise RuntimeError(
                "The ddgs library is required. Install with: pip install ddgs"
            )

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search DuckDuckGo for recent information.

        Args:
            query: Search term
            limit: Number of results to return

        Returns:
            List of search results (simplified format)
        """
        try:
            ddgs = DDGS(timeout=10)
            results_raw = list(ddgs.text(query, max_results=limit))
            
            if not results_raw:
                return []

            results = []
            for result in results_raw:
                results.append({
                    "title": result.get("title", ""),
                    "summary": result.get("body", ""),
                    "url": result.get("href", ""),
                    "source": "DuckDuckGo"
                })

            return results[:limit]

        except Exception as e:
            return [{
                "title": f"Search error",
                "summary": f"Failed to search for '{query}': {str(e)}",
                "source": "DuckDuckGo",
                "url": ""
            }]

    def news(self, query: str, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Recent news. ddgs only offers coarse windows: day / week / month / year."""
        timelimit = "d" if days <= 1 else "w" if days <= 7 else "m" if days <= 31 else "y"
        try:
            raw = DDGS(timeout=10).news(query, timelimit=timelimit, max_results=limit)
            return [
                {
                    "title": r.get("title", ""),
                    "summary": r.get("body", ""),
                    "url": r.get("url", ""),
                    "source": r.get("source", "") or "DuckDuckGo News",
                    "date": (r.get("date") or "")[:10],
                }
                for r in raw
            ][:limit]
        except Exception as e:
            return [{
                "title": "News search error",
                "summary": f"Failed to fetch news for '{query}': {str(e)}",
                "source": "DuckDuckGo News",
                "url": "",
            }]


class NewsAPIProvider(LiveDataProvider):
    """
    News search using NewsAPI.org (requires free API key).

    Get a free API key at: https://newsapi.org/register
    Set NEWS_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "NEWS_API_KEY environment variable not set. "
                "Get a free key at https://newsapi.org/register"
            )
        if requests is None:
            raise RuntimeError(
                "requests library is required for NewsAPIProvider. "
                "Install with: pip install requests"
            )
        self.base_url = "https://newsapi.org/v2/everything"
        # NewsAPI only searches news. General questions ("what is X?") go to
        # the web instead — when ddgs is installed.
        self._web = DuckDuckGoSearchProvider() if DDGS is not None else None

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """General web search — delegated, because NewsAPI can only find news
        articles that MENTION a topic, not pages that explain it."""
        if self._web is not None:
            return self._web.search(query, limit=limit)
        return self.news(query, days=30, limit=limit)

    def news(self, query: str, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for news articles published in the last `days` days.

        Args:
            query: Search term
            days:  Only articles from this many days back until today
            limit: Number of articles to return

        Returns:
            List of news articles (with "date")
        """
        try:
            # sortBy=relevancy, not publishedAt. This tool answers questions,
            # and publishedAt returns whatever matched most *recently* rather
            # than most *closely* — for "Christopher Nolan latest release" that
            # meant unrelated round-ups instead of the article naming the film.
            params = {
                "q": query,
                "sortBy": "relevancy",
                "apiKey": self.api_key,
                "pageSize": limit,
                # Without these two, "last 7 days" returned articles from any
                # date, in any language.
                "from": (date.today() - timedelta(days=days)).isoformat(),
                "language": "en",
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Surface the API's own failure reason rather than an empty list:
            # the agent's web_search tool reports an empty result as "no search
            # results found", which would misreport a quota/auth failure as the
            # topic simply having no coverage. Mirrors the except branch below.
            if data.get("status") != "ok":
                return [{
                    "title": "News search error",
                    "summary": (
                        f"NewsAPI returned an error for '{query}': "
                        f"{data.get('message', 'unknown error')}"
                    ),
                    "source": "NewsAPI",
                    "url": "",
                }]

            results = []
            for article in data.get("articles", [])[:limit]:
                results.append({
                    "title": article.get("title", ""),
                    "summary": article.get("description", "") or article.get("content", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", "NewsAPI"),
                    "date": (article.get("publishedAt") or "")[:10],
                })

            return results

        except Exception as e:
            return [{
                "title": f"News search error",
                "summary": f"Failed to fetch news for '{query}': {str(e)}",
                "source": "NewsAPI",
                "url": ""
            }]


class MockLiveDataProvider(LiveDataProvider):
    """Mock provider returning synthetic demo data. Used for testing/demo."""

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return demo data."""
        return [
            {
                "title": f"Mock result 1: {query}",
                "summary": f"This is a mock search result for '{query}'. In production, this would fetch real data.",
                "source": "Mock",
                "url": "https://example.com",
            },
            {
                "title": f"Mock result 2: {query}",
                "summary": f"Another mock result for '{query}'. The chatbot is currently in demo mode.",
                "source": "Mock",
                "url": "https://example.com",
            },
        ][:limit]


def create_live_data_provider(
    provider: Optional[str] = None,
    **kwargs,
) -> LiveDataProvider:
    """
    Factory for live data providers, selected by LIVE_DATA_PROVIDER env var.

    Args:
        provider: Provider name (overrides env var). Options:
                 - "duckduckgo" — web + news search via ddgs (no API key needed)
                 - "newsapi" — news via NewsAPI (NEWS_API_KEY), web via ddgs
                 - "mock" — returns synthetic data (for testing/demo)
        **kwargs: Additional arguments passed to the provider

    Returns:
        LiveDataProvider instance

    Raises:
        ValueError: If provider is unknown or misconfigured
    """
    provider = (provider or os.getenv("LIVE_DATA_PROVIDER", "mock")).strip().lower()

    if provider == "duckduckgo":
        return DuckDuckGoSearchProvider()

    if provider == "newsapi":
        api_key = kwargs.get("api_key")
        return NewsAPIProvider(api_key=api_key)

    if provider == "mock":
        return MockLiveDataProvider()

    raise ValueError(
        f"Unknown LIVE_DATA_PROVIDER={provider!r}. "
        "Supported: duckduckgo, newsapi, mock."
    )
