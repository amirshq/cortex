"""Tests for embedding providers."""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.business.core.embedding import Embedder, OpenAIEmbedder, create_embedder


class TestEmbedderInterface:
    """Test the Embedder abstract interface."""

    def test_cannot_instantiate_abstract_base(self):
        """Embedder is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            Embedder()

    def test_subclass_must_implement_methods(self):
        """Subclasses must implement required methods."""
        class IncompleteEmbedder(Embedder):
            pass

        with pytest.raises(TypeError):
            IncompleteEmbedder()


class TestOpenAIEmbedderInitialization:
    """Test OpenAIEmbedder initialization."""

    def test_initializes_with_api_key(self):
        """OpenAIEmbedder accepts api_key parameter."""
        with patch("src.business.core.embedding.OpenAI"):
            embedder = OpenAIEmbedder(api_key="test-key")
            assert embedder.model == "text-embedding-3-small"

    def test_initializes_with_custom_model(self):
        """OpenAIEmbedder accepts custom model parameter."""
        with patch("src.business.core.embedding.OpenAI"):
            embedder = OpenAIEmbedder(
                api_key="test-key",
                model="text-embedding-3-large"
            )
            assert embedder.model == "text-embedding-3-large"

    def test_accepts_preconfigured_client(self):
        """OpenAIEmbedder can use a pre-built client (e.g., Azure)."""
        mock_client = Mock()
        embedder = OpenAIEmbedder(client=mock_client)
        assert embedder.client is mock_client

    def test_creates_client_from_api_key(self):
        """OpenAIEmbedder creates client from api_key if not provided."""
        with patch("src.business.core.embedding.OpenAI") as mock_openai:
            embedder = OpenAIEmbedder(api_key="test-key")
            mock_openai.assert_called_once_with(api_key="test-key")


class TestOpenAIEmbedderEmbedQuery:
    """Test embed_query method."""

    def test_embed_single_query(self):
        """embed_query embeds a single text."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(client=mock_client)
        result = embedder.embed_query("Hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    def test_embed_query_is_list(self):
        """embed_query returns a list of floats."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.5] * 1536)]  # 1536-dim embedding
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(client=mock_client)
        result = embedder.embed_query("test")

        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)

    def test_embed_query_uses_correct_model(self):
        """embed_query uses the configured model."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1])]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(client=mock_client, model="text-embedding-3-large")
        embedder.embed_query("test")

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-large"


class TestOpenAIEmbedderEmbedDocuments:
    """Test embed_documents method."""

    def test_embed_empty_list(self):
        """embed_documents returns empty list for empty input."""
        mock_client = Mock()
        embedder = OpenAIEmbedder(client=mock_client)
        result = embedder.embed_documents([])
        assert result == []
        mock_client.embeddings.create.assert_not_called()

    def test_embed_multiple_documents(self):
        """embed_documents embeds multiple texts."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4]),
            Mock(embedding=[0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(client=mock_client)
        result = embedder.embed_documents(["text1", "text2", "text3"])

        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
        assert result[2] == [0.5, 0.6]

    def test_embed_documents_batches_correctly(self):
        """embed_documents passes all texts in a single batch."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1]) for _ in range(5)]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(client=mock_client)
        texts = ["text1", "text2", "text3", "text4", "text5"]
        embedder.embed_documents(texts)

        # Verify client was called with all texts
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == texts


class TestOpenAIEmbedderBackwardCompatibility:
    """Test backward compatibility methods."""

    def test_embed_method_calls_embed_query(self):
        """embed() is an alias for embed_query() (backward compat)."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(client=mock_client)
        result = embedder.embed("test")

        assert result == [0.1, 0.2]


class TestOpenAIEmbedderRetryLogic:
    """Test retry logic for transient errors."""

    def test_retries_on_internal_server_error(self):
        """_embed retries on InternalServerError."""
        from openai import InternalServerError

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1])]

        # Create mock response object for InternalServerError
        mock_http_response = Mock()
        mock_http_response.status_code = 500

        # First call fails, second succeeds
        mock_client.embeddings.create.side_effect = [
            InternalServerError(message="500 error", response=mock_http_response, body={"error": "500"}),
            mock_response,
        ]

        embedder = OpenAIEmbedder(client=mock_client)
        with patch("time.sleep"):  # Don't actually sleep in tests
            result = embedder.embed_query("test")

        assert result == [0.1]
        assert mock_client.embeddings.create.call_count == 2

    def test_gives_up_after_max_retries(self):
        """_embed gives up after max_retries."""
        from openai import InternalServerError

        mock_client = Mock()
        mock_http_response = Mock()
        mock_http_response.status_code = 500

        mock_client.embeddings.create.side_effect = InternalServerError(
            message="500 error",
            response=mock_http_response,
            body={"error": "500"}
        )

        embedder = OpenAIEmbedder(client=mock_client)
        with patch("time.sleep"):
            with pytest.raises(InternalServerError):
                embedder.embed_query("test")

        # Should have tried max_retries times (5 by default)
        assert mock_client.embeddings.create.call_count == 5

    def test_retry_delay_increases_exponentially(self):
        """Retry delay increases exponentially (2^attempt)."""
        from openai import InternalServerError

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1])]

        mock_http_response = Mock()
        mock_http_response.status_code = 500

        mock_client.embeddings.create.side_effect = [
            InternalServerError(message="500", response=mock_http_response, body={"error": "500"}),
            InternalServerError(message="500", response=mock_http_response, body={"error": "500"}),
            mock_response,
        ]

        embedder = OpenAIEmbedder(client=mock_client)

        sleep_calls = []
        with patch("time.sleep", side_effect=lambda x: sleep_calls.append(x)):
            result = embedder.embed_query("test")

        # Should have 2 retries with delays 2^0=1, 2^1=2
        assert sleep_calls == [1, 2]


class TestCreateEmbedderFactory:
    """Test the create_embedder factory function."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_default_provider_is_openai(self):
        """Default provider is OpenAI."""
        embedder = create_embedder()
        assert isinstance(embedder, OpenAIEmbedder)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_explicit_openai_provider(self):
        """provider='openai' creates OpenAIEmbedder."""
        embedder = create_embedder(provider="openai")
        assert isinstance(embedder, OpenAIEmbedder)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_custom_model_name(self):
        """model parameter is passed through."""
        embedder = create_embedder(
            provider="openai",
            model="text-embedding-3-large"
        )
        assert embedder.model == "text-embedding-3-large"

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_openai_requires_api_key(self):
        """OpenAI provider raises error without OPENAI_API_KEY."""
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY must be set"):
            create_embedder(provider="openai")

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "test-embedding"
    })
    def test_azure_openai_provider(self):
        """provider='azure_openai' creates OpenAIEmbedder with Azure client."""
        embedder = create_embedder(provider="azure_openai")
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.model == "test-embedding"

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key"
    }, clear=False)
    def test_azure_requires_embedding_deployment_name(self):
        """Azure requires AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME."""
        with patch.dict(os.environ, {"AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": ""}, clear=False):
            with pytest.raises(RuntimeError, match="AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"):
                create_embedder(provider="azure_openai")

    def test_unknown_provider_raises_error(self):
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
            create_embedder(provider="unknown_provider")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_case_insensitive_provider(self):
        """Provider name is case-insensitive."""
        embedder1 = create_embedder(provider="OPENAI")
        embedder2 = create_embedder(provider="OpenAI")
        assert isinstance(embedder1, OpenAIEmbedder)
        assert isinstance(embedder2, OpenAIEmbedder)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_env_var_provider_selection(self):
        """EMBEDDING_PROVIDER env var selects provider."""
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai"}):
            embedder = create_embedder()
            assert isinstance(embedder, OpenAIEmbedder)
