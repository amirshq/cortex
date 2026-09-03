"""Tests for live data providers."""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.business.core.live_data import (
    LiveDataProvider,
    DuckDuckGoSearchProvider,
    NewsAPIProvider,
    MockLiveDataProvider,
    create_live_data_provider,
)


class TestLiveDataProviderInterface:
    """Test the LiveDataProvider abstract interface."""

    def test_cannot_instantiate_abstract_base(self):
        """LiveDataProvider is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            LiveDataProvider()

    def test_subclass_must_implement_search(self):
        """Subclasses must implement search()."""
        class IncompleteProvider(LiveDataProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()


class TestMockLiveDataProvider:
    """Test mock provider (safe for testing without API calls)."""

    def test_search_returns_results(self):
        """Mock provider returns synthetic results."""
        provider = MockLiveDataProvider()
        results = provider.search("test query")

        assert len(results) > 0
        assert all("title" in r for r in results)
        assert all("summary" in r for r in results)
        assert all("source" in r for r in results)

    def test_search_respects_limit(self):
        """Mock provider respects limit parameter."""
        provider = MockLiveDataProvider()
        results = provider.search("test", limit=1)

        assert len(results) <= 1

    def test_search_returns_dict_structure(self):
        """Results have expected structure."""
        provider = MockLiveDataProvider()
        results = provider.search("test")

        for result in results:
            assert isinstance(result, dict)
            assert "title" in result
            assert "summary" in result
            assert "source" in result

    def test_search_includes_query_in_results(self):
        """Query term appears in mock results."""
        provider = MockLiveDataProvider()
        query = "special test query"
        results = provider.search(query)

        # At least one result should mention the query
        result_text = " ".join(r.get("title", "") + r.get("summary", "") for r in results)
        assert query.lower() in result_text.lower()


class TestDuckDuckGoSearchProvider:
    """Test DuckDuckGo search provider."""

    def test_initialization_checks_requests(self):
        """DuckDuckGo provider requires requests library."""
        with patch("src.business.core.live_data.requests", None):
            with pytest.raises(RuntimeError, match="requests"):
                DuckDuckGoSearchProvider()

    def test_search_makes_api_call(self):
        """search() calls DuckDuckGo API."""
        provider = DuckDuckGoSearchProvider()

        mock_response = Mock()
        mock_response.json.return_value = {
            "Heading": "Test",
            "AbstractText": "Test abstract",
            "AbstractURL": "https://example.com",
            "RelatedTopics": [
                {"Text": "Result 1", "FirstURL": "https://example.com/1"},
                {"Text": "Result 2", "FirstURL": "https://example.com/2"},
            ]
        }

        with patch("src.business.core.live_data.requests.get", return_value=mock_response):
            results = provider.search("test query")

        assert len(results) > 0
        assert all("title" in r for r in results)
        assert all("summary" in r for r in results)

    def test_search_handles_api_error(self):
        """search() handles API errors gracefully."""
        provider = DuckDuckGoSearchProvider()

        with patch("src.business.core.live_data.requests.get", side_effect=Exception("API error")):
            results = provider.search("test")

        assert len(results) > 0
        assert "error" in results[0]["summary"].lower() or "failed" in results[0]["summary"].lower()

    def test_search_respects_limit(self):
        """search() respects limit parameter."""
        provider = DuckDuckGoSearchProvider()

        mock_response = Mock()
        mock_response.json.return_value = {
            "Heading": "Test",
            "AbstractText": "Abstract",
            "RelatedTopics": [
                {"Text": f"Result {i}", "FirstURL": f"https://example.com/{i}"}
                for i in range(10)
            ]
        }

        with patch("src.business.core.live_data.requests.get", return_value=mock_response):
            results = provider.search("test", limit=3)

        assert len(results) <= 3


class TestNewsAPIProvider:
    """Test NewsAPI provider."""

    def test_initialization_requires_api_key(self):
        """NewsAPI provider requires API key."""
        with patch.dict(os.environ, {"NEWS_API_KEY": ""}, clear=False):
            with pytest.raises(RuntimeError, match="NEWS_API_KEY"):
                NewsAPIProvider()

    def test_initialization_accepts_api_key_parameter(self):
        """NewsAPI accepts api_key parameter."""
        provider = NewsAPIProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_initialization_from_env_var(self):
        """NewsAPI reads API key from environment."""
        with patch.dict(os.environ, {"NEWS_API_KEY": "env-key"}):
            provider = NewsAPIProvider()
            assert provider.api_key == "env-key"

    def test_initialization_prefers_parameter_over_env(self):
        """NewsAPI prefers parameter api_key over env var."""
        with patch.dict(os.environ, {"NEWS_API_KEY": "env-key"}):
            provider = NewsAPIProvider(api_key="param-key")
            assert provider.api_key == "param-key"

    def test_initialization_checks_requests(self):
        """NewsAPI provider requires requests library."""
        with patch.dict(os.environ, {"NEWS_API_KEY": "test"}):
            with patch("src.business.core.live_data.requests", None):
                with pytest.raises(RuntimeError, match="requests"):
                    NewsAPIProvider()

    def test_search_makes_api_call(self):
        """search() calls NewsAPI."""
        with patch.dict(os.environ, {"NEWS_API_KEY": "test-key"}):
            provider = NewsAPIProvider()

            mock_response = Mock()
            mock_response.json.return_value = {
                "status": "ok",
                "articles": [
                    {
                        "title": "Test Article",
                        "description": "Test description",
                        "source": {"name": "Test Source"},
                        "url": "https://example.com",
                        "publishedAt": "2024-01-01T00:00:00Z"
                    }
                ]
            }

            with patch("src.business.core.live_data.requests.get", return_value=mock_response):
                results = provider.search("test")

            assert len(results) == 1
            assert results[0]["title"] == "Test Article"

    def test_search_handles_api_error(self):
        """search() handles API errors."""
        with patch.dict(os.environ, {"NEWS_API_KEY": "test-key"}):
            provider = NewsAPIProvider()

            with patch("src.business.core.live_data.requests.get", side_effect=Exception("API error")):
                results = provider.search("test")

            assert len(results) > 0
            assert "error" in results[0]["summary"].lower() or "failed" in results[0]["summary"].lower()

    def test_search_handles_api_error_response(self):
        """search() handles API error responses."""
        with patch.dict(os.environ, {"NEWS_API_KEY": "test-key"}):
            provider = NewsAPIProvider()

            mock_response = Mock()
            mock_response.json.return_value = {
                "status": "error",
                "message": "API quota exceeded"
            }

            with patch("src.business.core.live_data.requests.get", return_value=mock_response):
                results = provider.search("test")

            assert len(results) > 0
            assert "error" in results[0]["summary"].lower()

    def test_search_respects_limit(self):
        """search() respects limit parameter."""
        with patch.dict(os.environ, {"NEWS_API_KEY": "test-key"}):
            provider = NewsAPIProvider()

            mock_response = Mock()
            mock_response.json.return_value = {
                "status": "ok",
                "articles": [
                    {
                        "title": f"Article {i}",
                        "description": "Description",
                        "source": {"name": "Source"},
                        "url": "https://example.com"
                    }
                    for i in range(10)
                ]
            }

            with patch("src.business.core.live_data.requests.get", return_value=mock_response):
                results = provider.search("test", limit=3)

            assert len(results) == 3


class TestCreateLiveDataProviderFactory:
    """Test create_live_data_provider factory."""

    def test_default_provider_is_mock(self):
        """Default provider is mock."""
        provider = create_live_data_provider()
        assert isinstance(provider, MockLiveDataProvider)

    def test_explicit_mock_provider(self):
        """provider='mock' creates MockLiveDataProvider."""
        provider = create_live_data_provider(provider="mock")
        assert isinstance(provider, MockLiveDataProvider)

    def test_duckduckgo_provider(self):
        """provider='duckduckgo' creates DuckDuckGoSearchProvider."""
        provider = create_live_data_provider(provider="duckduckgo")
        assert isinstance(provider, DuckDuckGoSearchProvider)

    @patch.dict(os.environ, {"NEWS_API_KEY": "test-key"})
    def test_newsapi_provider(self):
        """provider='newsapi' creates NewsAPIProvider."""
        provider = create_live_data_provider(provider="newsapi")
        assert isinstance(provider, NewsAPIProvider)

    def test_env_var_provider_selection(self):
        """LIVE_DATA_PROVIDER env var selects provider."""
        with patch.dict(os.environ, {"LIVE_DATA_PROVIDER": "mock"}):
            provider = create_live_data_provider()
            assert isinstance(provider, MockLiveDataProvider)

    def test_parameter_overrides_env_var(self):
        """provider parameter overrides env var."""
        with patch.dict(os.environ, {"LIVE_DATA_PROVIDER": "duckduckgo"}):
            provider = create_live_data_provider(provider="mock")
            assert isinstance(provider, MockLiveDataProvider)

    def test_case_insensitive_provider_name(self):
        """Provider name is case-insensitive."""
        provider1 = create_live_data_provider(provider="MOCK")
        provider2 = create_live_data_provider(provider="Mock")
        assert isinstance(provider1, MockLiveDataProvider)
        assert isinstance(provider2, MockLiveDataProvider)

    def test_unknown_provider_raises_error(self):
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            create_live_data_provider(provider="unknown_provider")

    def test_passes_kwargs_to_newsapi(self):
        """kwargs are passed to NewsAPIProvider."""
        with patch("src.business.core.live_data.NewsAPIProvider") as mock_newsapi:
            create_live_data_provider(provider="newsapi", api_key="custom-key")
            mock_newsapi.assert_called_once_with(api_key="custom-key")
