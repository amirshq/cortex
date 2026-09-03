"""Tests for cost calculation."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.business.core.cost import (
    calculate_chat_cost,
    calculate_embedding_cost,
    get_model_pricing,
    get_embedding_pricing,
    ModelPricing,
    EmbeddingPricing,
)


class TestCalculateChatCost:
    """Test chat completion cost calculation."""

    def test_calculate_cost_gpt_4o_openai(self):
        """Calculate cost for GPT-4o on OpenAI."""
        cost = calculate_chat_cost(
            model_name="gpt-4o",
            input_tokens=1_000_000,  # 1M tokens
            output_tokens=1_000_000,
            provider="openai"
        )
        # gpt-4o: $5/1M input, $15/1M output = $20 total
        assert cost == 20.0

    def test_calculate_cost_gpt_3_5_turbo(self):
        """Calculate cost for GPT-3.5-turbo."""
        cost = calculate_chat_cost(
            model_name="gpt-3.5-turbo",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            provider="openai"
        )
        # gpt-3.5-turbo: $0.5/1M input, $1.5/1M output = $2 total
        assert cost == 2.0

    def test_calculate_cost_partial_tokens(self):
        """Calculate cost with partial token counts."""
        cost = calculate_chat_cost(
            model_name="gpt-4o",
            input_tokens=100_000,  # 0.1M
            output_tokens=50_000,  # 0.05M
            provider="openai"
        )
        # input: 100k / 1M * 5 = 0.5
        # output: 50k / 1M * 15 = 0.75
        # total: 1.25
        assert cost == 1.25

    def test_calculate_cost_zero_output_tokens(self):
        """Calculate cost with only input tokens."""
        cost = calculate_chat_cost(
            model_name="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=0,
            provider="openai"
        )
        assert cost == 5.0  # Only input cost

    def test_calculate_cost_zero_input_tokens(self):
        """Calculate cost with only output tokens."""
        cost = calculate_chat_cost(
            model_name="gpt-4o",
            input_tokens=0,
            output_tokens=1_000_000,
            provider="openai"
        )
        assert cost == 15.0  # Only output cost

    def test_calculate_cost_unknown_model(self):
        """Unknown model returns 0.0 cost."""
        cost = calculate_chat_cost(
            model_name="unknown-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            provider="openai"
        )
        assert cost == 0.0

    def test_calculate_cost_azure_openai(self):
        """Calculate cost for Azure OpenAI."""
        cost = calculate_chat_cost(
            model_name="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            provider="azure_openai"
        )
        # Same pricing as OpenAI for gpt-4o
        assert cost == 20.0

    def test_cost_precision(self):
        """Cost is rounded to 6 decimal places."""
        cost = calculate_chat_cost(
            model_name="gpt-3.5-turbo",
            input_tokens=1,
            output_tokens=1,
            provider="openai"
        )
        # 1/1M * 0.5 + 1/1M * 1.5 = 0.000002
        assert cost == 0.000002

    def test_calculate_cost_large_numbers(self):
        """Calculate cost with large token counts."""
        cost = calculate_chat_cost(
            model_name="gpt-4o",
            input_tokens=10_000_000,  # 10M
            output_tokens=5_000_000,  # 5M
            provider="openai"
        )
        # input: 10M / 1M * 5 = 50
        # output: 5M / 1M * 15 = 75
        # total: 125
        assert cost == 125.0


class TestCalculateEmbeddingCost:
    """Test embedding cost calculation."""

    def test_calculate_embedding_cost_small(self):
        """Calculate cost for text-embedding-3-small."""
        cost = calculate_embedding_cost(
            num_tokens=1_000_000,
            model_name="text-embedding-3-small",
            provider="openai"
        )
        # $0.02 per 1M tokens
        assert cost == 0.02

    def test_calculate_embedding_cost_large(self):
        """Calculate cost for text-embedding-3-large."""
        cost = calculate_embedding_cost(
            num_tokens=1_000_000,
            model_name="text-embedding-3-large",
            provider="openai"
        )
        # $0.13 per 1M tokens
        assert cost == 0.13

    def test_calculate_embedding_cost_partial(self):
        """Calculate cost with partial token count."""
        cost = calculate_embedding_cost(
            num_tokens=100_000,  # 0.1M
            model_name="text-embedding-3-small",
            provider="openai"
        )
        # 0.1M / 1M * 0.02 = 0.002
        assert cost == 0.002

    def test_calculate_embedding_cost_zero(self):
        """Calculate cost with zero tokens."""
        cost = calculate_embedding_cost(
            num_tokens=0,
            model_name="text-embedding-3-small",
            provider="openai"
        )
        assert cost == 0.0

    def test_calculate_embedding_cost_unknown_model(self):
        """Unknown embedding model returns 0.0 cost."""
        cost = calculate_embedding_cost(
            num_tokens=1_000_000,
            model_name="unknown-embedding",
            provider="openai"
        )
        assert cost == 0.0

    def test_calculate_embedding_cost_azure(self):
        """Calculate cost for Azure embedding."""
        cost = calculate_embedding_cost(
            num_tokens=1_000_000,
            model_name="text-embedding-3-small",
            provider="azure_openai"
        )
        # Same pricing as OpenAI
        assert cost == 0.02

    def test_embedding_cost_precision(self):
        """Embedding cost is rounded to 6 decimal places."""
        cost = calculate_embedding_cost(
            num_tokens=1,
            model_name="text-embedding-3-small",
            provider="openai"
        )
        # 1/1M * 0.02 = 0.00000002, rounded to 6 decimals = 0.0
        assert cost == 0.0


class TestGetModelPricing:
    """Test get_model_pricing function."""

    def test_get_pricing_openai(self):
        """Get pricing for OpenAI model."""
        pricing = get_model_pricing("gpt-4o", provider="openai")
        assert pricing is not None
        assert pricing.model_name == "gpt-4o"
        assert pricing.input_tokens_per_1m == 5.0
        assert pricing.output_tokens_per_1m == 15.0

    def test_get_pricing_azure(self):
        """Get pricing for Azure OpenAI model."""
        pricing = get_model_pricing("gpt-4o", provider="azure_openai")
        assert pricing is not None
        assert pricing.model_name == "gpt-4o"

    def test_get_pricing_unknown_model(self):
        """Get pricing for unknown model returns None."""
        pricing = get_model_pricing("unknown-model", provider="openai")
        assert pricing is None

    def test_get_pricing_default_provider(self):
        """Default provider is OpenAI."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert pricing.input_tokens_per_1m == 5.0


class TestGetEmbeddingPricing:
    """Test get_embedding_pricing function."""

    def test_get_embedding_pricing_small(self):
        """Get pricing for small embedding model."""
        pricing = get_embedding_pricing("text-embedding-3-small", provider="openai")
        assert pricing is not None
        assert pricing.model_name == "text-embedding-3-small"
        assert pricing.tokens_per_1m == 0.02

    def test_get_embedding_pricing_large(self):
        """Get pricing for large embedding model."""
        pricing = get_embedding_pricing("text-embedding-3-large", provider="openai")
        assert pricing is not None
        assert pricing.tokens_per_1m == 0.13

    def test_get_embedding_pricing_unknown(self):
        """Get pricing for unknown model returns None."""
        pricing = get_embedding_pricing("unknown-embedding", provider="openai")
        assert pricing is None

    def test_get_embedding_pricing_azure(self):
        """Get pricing for Azure embedding model."""
        pricing = get_embedding_pricing("text-embedding-3-small", provider="azure_openai")
        assert pricing is not None


class TestModelPricingDataclass:
    """Test ModelPricing dataclass."""

    def test_model_pricing_creation(self):
        """Create ModelPricing instance."""
        pricing = ModelPricing(
            model_name="test-model",
            input_tokens_per_1m=1.0,
            output_tokens_per_1m=2.0
        )
        assert pricing.model_name == "test-model"
        assert pricing.input_tokens_per_1m == 1.0
        assert pricing.output_tokens_per_1m == 2.0


class TestEmbeddingPricingDataclass:
    """Test EmbeddingPricing dataclass."""

    def test_embedding_pricing_creation(self):
        """Create EmbeddingPricing instance."""
        pricing = EmbeddingPricing(
            model_name="test-embedding",
            tokens_per_1m=0.5
        )
        assert pricing.model_name == "test-embedding"
        assert pricing.tokens_per_1m == 0.5
