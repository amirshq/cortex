"""Live data providers for real-time information (news, weather, web search, etc.)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import json

try:
    import requests
except ImportError:
    requests = None

try:
    from duckduckgo_search import DDGS
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


class DuckDuckGoSearchProvider(LiveDataProvider):
    """
    Web search using DuckDuckGo via duckduckgo-search library (free, no API key required).
    
    This uses the community-maintained duckduckgo-search package which is more reliable
    than the public API.
    """

    def __init__(self):
        if DDGS is None:
            raise RuntimeError(
                "duckduckgo-search library is required. "
                "Install with: pip install duckduckgo-search"
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

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for recent news articles.

        Args:
            query: Search term
            limit: Number of articles to return

        Returns:
            List of news articles
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
                    "source": article.get("source", {}).get("name", "NewsAPI")
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
                 - "duckduckgo" — free web search (no API key needed)
                 - "newsapi" — news search (requires NEWS_API_KEY env var)
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
