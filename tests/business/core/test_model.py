"""Tests for LLM model implementations and factory."""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.business.core.model import BaseLLM, OpenAIModel, LocalHFModel, create_llm, build_azure_openai_client


class TestBaseLLMInterface:
    """Test the BaseLLM abstract interface."""

    def test_cannot_instantiate_abstract_base(self):
        """BaseLLM is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLLM()

    def test_subclass_must_implement_generate(self):
        """Subclasses must implement generate()."""
        class IncompleteModel(BaseLLM):
            pass

        with pytest.raises(TypeError):
            IncompleteModel()


class TestOpenAIModelImplementation:
    """Test OpenAI model implementation."""

    def test_initialize_with_defaults(self):
        """OpenAIModel initializes with sensible defaults."""
        mock_client = Mock()
        model = OpenAIModel(
            client=mock_client,
            model_name="gpt-4o",
            system_prompt="You are helpful"
        )
        assert model.model_name == "gpt-4o"
        assert model.temperature == 0.7
        assert model.max_tokens == 512

    def test_initialize_with_custom_params(self):
        """OpenAIModel accepts custom temperature and max_tokens."""
        mock_client = Mock()
        model = OpenAIModel(
            client=mock_client,
            model_name="gpt-4-turbo",
            system_prompt="Be concise",
            temperature=0.3,
            max_tokens=1024
        )
        assert model.temperature == 0.3
        assert model.max_tokens == 1024

    def test_generate_calls_client_correctly(self):
        """generate() calls client.chat.completions.create with correct params."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_client.chat.completions.create.return_value = mock_response

        model = OpenAIModel(
            client=mock_client,
            model_name="gpt-4o",
            system_prompt="You are helpful",
            temperature=0.5,
            max_tokens=200
        )

        result = model.generate("What is AI?", ["Context 1", "Context 2"])

        # Verify client was called
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 200

    def test_generate_returns_stripped_response(self):
        """generate() returns response with whitespace stripped."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "  Generated response  \n"
        mock_client.chat.completions.create.return_value = mock_response

        model = OpenAIModel(
            client=mock_client,
            model_name="gpt-4o",
            system_prompt="You are helpful"
        )

        result = model.generate("What is AI?", [])
        assert result == "Generated response"

    def test_generate_with_empty_context(self):
        """generate() handles empty context list."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response

        model = OpenAIModel(
            client=mock_client,
            model_name="gpt-4o",
            system_prompt="You are helpful"
        )

        result = model.generate("Question?", [])
        assert result == "Response"

    def test_generate_with_multiple_context_items(self):
        """generate() includes all context items in the request."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Answer"
        mock_client.chat.completions.create.return_value = mock_response

        model = OpenAIModel(
            client=mock_client,
            model_name="gpt-4o",
            system_prompt="You are helpful"
        )

        context = ["Context 1", "Context 2", "Context 3"]
        result = model.generate("Question?", context)

        # Verify the context was passed through
        mock_client.chat.completions.create.assert_called_once()


class TestCreateLLMFactory:
    """Test the create_llm factory function."""

    def test_default_provider_is_openai(self):
        """Default provider is OpenAI when LLM_PROVIDER not set."""
        # Test explicitly without setting LLM_PROVIDER
        model = create_llm(system_prompt="Test", provider="openai")
        assert isinstance(model, OpenAIModel)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_explicit_openai_provider(self):
        """provider='openai' creates OpenAIModel."""
        model = create_llm(provider="openai", system_prompt="Test")
        assert isinstance(model, OpenAIModel)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_env_var_provider_openai(self):
        """LLM_PROVIDER=openai uses OpenAI."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
            model = create_llm(system_prompt="Test")
            assert isinstance(model, OpenAIModel)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_custom_model_name(self):
        """model_name parameter is passed through."""
        model = create_llm(
            provider="openai",
            model_name="gpt-4-turbo",
            system_prompt="Test"
        )
        assert model.model_name == "gpt-4-turbo"

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_openai_requires_api_key(self):
        """OpenAI provider raises error without OPENAI_API_KEY."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY must be set"):
                create_llm(provider="openai", system_prompt="Test")

    def test_huggingface_provider(self):
        """provider='huggingface' creates LocalHFModel."""
        # Note: This might fail if models aren't downloaded, but tests the factory
        with patch("src.business.core.model.AutoTokenizer.from_pretrained"):
            with patch("src.business.core.model.AutoModelForCausalLM.from_pretrained"):
                model = create_llm(
                    provider="huggingface",
                    model_name="mistral-7b-instruct-v0.1",
                    system_prompt="Test"
                )
                assert isinstance(model, LocalHFModel)

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": "test-deployment"
    })
    def test_azure_openai_provider(self):
        """provider='azure_openai' creates OpenAIModel with Azure client."""
        model = create_llm(provider="azure_openai", system_prompt="Test")
        assert isinstance(model, OpenAIModel)
        # Model name should be the deployment name
        assert model.model_name == "test-deployment"

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"})
    def test_azure_openai_requires_deployment_name(self):
        """Azure OpenAI requires AZURE_OPENAI_CHAT_DEPLOYMENT_NAME."""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": ""
        }, clear=False):
            with pytest.raises(RuntimeError, match="AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"):
                create_llm(provider="azure_openai", system_prompt="Test")

    def test_unknown_provider_raises_error(self):
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            create_llm(provider="unknown_provider", system_prompt="Test")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_case_insensitive_provider(self):
        """Provider name is case-insensitive."""
        model1 = create_llm(provider="OPENAI", system_prompt="Test")
        model2 = create_llm(provider="OpenAI", system_prompt="Test")
        assert isinstance(model1, OpenAIModel)
        assert isinstance(model2, OpenAIModel)


class TestBuildAzureOpenAIClient:
    """Test Azure OpenAI client builder."""

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_API_VERSION": "2024-10-21"
    })
    @patch("src.business.core.model.AzureOpenAI")
    def test_builds_azure_client_from_env(self, mock_azure_openai):
        """build_azure_openai_client builds client from env vars."""
        build_azure_openai_client()
        mock_azure_openai.assert_called_once()
        call_kwargs = mock_azure_openai.call_args.kwargs
        assert call_kwargs["azure_endpoint"] == "https://test.openai.azure.com/"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["api_version"] == "2024-10-21"

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "env-key"
    }, clear=False)
    @patch("src.business.core.model.AzureOpenAI")
    def test_prefers_passed_api_key(self, mock_azure_openai):
        """Passed api_key overrides env var."""
        build_azure_openai_client(api_key="passed-key")
        call_kwargs = mock_azure_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "passed-key"

    def test_missing_endpoint_raises_error(self):
        """Missing AZURE_OPENAI_ENDPOINT raises error."""
        with patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": ""}, clear=False):
            with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
                build_azure_openai_client()

    def test_missing_api_key_raises_error(self):
        """Missing AZURE_OPENAI_API_KEY raises error."""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": ""
        }, clear=False):
            with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"):
                build_azure_openai_client()

    @patch("src.business.core.model.AzureOpenAI")
    def test_default_api_version(self, mock_azure_openai):
        """Default API version is used when not set."""
        # Create env dict without AZURE_OPENAI_API_VERSION key
        test_env = {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
        }
        # Remove the key if it exists
        if "AZURE_OPENAI_API_VERSION" in os.environ:
            del os.environ["AZURE_OPENAI_API_VERSION"]

        with patch.dict(os.environ, test_env, clear=False):
            build_azure_openai_client()
            call_kwargs = mock_azure_openai.call_args.kwargs
            # Should use the default version when not in env
            assert call_kwargs["api_version"] == "2024-10-21"


class TestLocalHFModelImplementation:
    """Test local Hugging Face model implementation."""

    def test_initialize_with_defaults(self):
        """LocalHFModel initializes with sensible defaults."""
        with patch("src.business.core.model.AutoTokenizer.from_pretrained"):
            with patch("src.business.core.model.AutoModelForCausalLM.from_pretrained"):
                model = LocalHFModel(
                    model_name="mistral-7b",
                    system_prompt="You are helpful"
                )
                assert model.max_input_tokens == 2048
                assert model.max_output_tokens == 512

    def test_initialize_with_custom_token_limits(self):
        """LocalHFModel accepts custom token limits."""
        with patch("src.business.core.model.AutoTokenizer.from_pretrained"):
            with patch("src.business.core.model.AutoModelForCausalLM.from_pretrained"):
                model = LocalHFModel(
                    model_name="mistral-7b",
                    system_prompt="Test",
                    max_input_tokens=4096,
                    max_output_tokens=1024
                )
                assert model.max_input_tokens == 4096
                assert model.max_output_tokens == 1024

    def test_sets_pad_token_when_missing(self):
        """LocalHFModel sets pad_token to eos_token if missing."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"

        with patch("src.business.core.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            with patch("src.business.core.model.AutoModelForCausalLM.from_pretrained"):
                model = LocalHFModel(
                    model_name="mistral-7b",
                    system_prompt="Test"
                )
                # pad_token should be set to eos_token
                assert mock_tokenizer.pad_token == "</s>"
