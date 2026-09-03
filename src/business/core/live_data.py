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
    Web search using DuckDuckGo API (free, no API key required).

    Note: This is a simple implementation using the public API.
    For production use, consider official API or alternatives.
    """

    def __init__(self):
        if requests is None:
            raise RuntimeError(
                "requests library is required for DuckDuckGoSearchProvider. "
                "Install with: pip install requests"
            )
        self.base_url = "https://api.duckduckgo.com"

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
            params = {
                "q": query,
                "format": "json",
                "no_redirect": 1,
            }

            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            results = []

            # Extract from related searches (if available)
            if "RelatedTopics" in data:
                for item in data["RelatedTopics"][:limit]:
                    if isinstance(item, dict) and "Text" in item:
                        results.append({
                            "title": item.get("Text", "")[:200],
                            "summary": item.get("Text", ""),
                            "url": item.get("FirstURL", ""),
                            "source": "DuckDuckGo"
                        })

            # Also include the abstract if available
            if "AbstractText" in data and data["AbstractText"] and len(results) < limit:
                results.insert(0, {
                    "title": data.get("Heading", query),
                    "summary": data["AbstractText"],
                    "url": data.get("AbstractURL", ""),
                    "source": "DuckDuckGo"
                })

            return results[:limit]

        except Exception as e:
            return [{
                "title": f"Search failed: {str(e)}",
                "summary": f"Unable to search for '{query}': {str(e)}",
                "source": "DuckDuckGo"
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
            params = {
                "q": query,
                "sortBy": "publishedAt",
                "apiKey": self.api_key,
                "pageSize": limit,
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                return [{
                    "title": "News search unavailable",
                    "summary": f"NewsAPI error: {data.get('message', 'Unknown error')}",
                    "source": "NewsAPI"
                }]

            results = []
            for article in data.get("articles", [])[:limit]:
                results.append({
                    "title": article.get("title", ""),
                    "summary": article.get("description", ""),
                    "source": article.get("source", {}).get("name", "NewsAPI"),
                    "url": article.get("url", ""),
                    "published": article.get("publishedAt", ""),
                })

            return results

        except Exception as e:
            return [{
                "title": f"News search failed: {str(e)}",
                "summary": f"Unable to search for '{query}': {str(e)}",
                "source": "NewsAPI"
            }]


class MockLiveDataProvider(LiveDataProvider):
    """
    Mock provider for testing (no external API calls).
    Returns synthetic data matching the search query.
    """

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return mock search results."""
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
