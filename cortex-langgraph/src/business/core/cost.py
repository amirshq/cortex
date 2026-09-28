"""Cost calculation for LLM and embedding API calls.

Pricing models are based on OpenAI's public pricing (https://openai.com/pricing/),
updated periodically. Azure OpenAI pricing is similar but may vary by region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelPricing:
    """Pricing for a specific LLM model."""
    model_name: str
    input_tokens_per_1m: float  # Cost per 1M input tokens in USD
    output_tokens_per_1m: float  # Cost per 1M output tokens in USD


@dataclass
class EmbeddingPricing:
    """Pricing for embedding model."""
    model_name: str
    tokens_per_1m: float  # Cost per 1M tokens in USD


# OpenAI pricing (as of 2024-09, subject to change)
OPENAI_MODELS = {
    "gpt-4o": ModelPricing("gpt-4o", input_tokens_per_1m=5.0, output_tokens_per_1m=15.0),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", input_tokens_per_1m=10.0, output_tokens_per_1m=30.0),
    "gpt-4": ModelPricing("gpt-4", input_tokens_per_1m=30.0, output_tokens_per_1m=60.0),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", input_tokens_per_1m=0.5, output_tokens_per_1m=1.5),
}

EMBEDDING_MODELS = {
    "text-embedding-3-small": EmbeddingPricing("text-embedding-3-small", tokens_per_1m=0.02),
    "text-embedding-3-large": EmbeddingPricing("text-embedding-3-large", tokens_per_1m=0.13),
    "text-embedding-ada-002": EmbeddingPricing("text-embedding-ada-002", tokens_per_1m=0.10),
}

# Azure OpenAI pricing (as of 2024, region-dependent; using US East pricing)
AZURE_OPENAI_MODELS = {
    "gpt-4o": ModelPricing("gpt-4o", input_tokens_per_1m=5.0, output_tokens_per_1m=15.0),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", input_tokens_per_1m=10.0, output_tokens_per_1m=30.0),
    "gpt-35-turbo": ModelPricing("gpt-35-turbo", input_tokens_per_1m=0.5, output_tokens_per_1m=1.5),
}

AZURE_EMBEDDING_MODELS = {
    "text-embedding-3-small": EmbeddingPricing("text-embedding-3-small", tokens_per_1m=0.02),
    "text-embedding-3-large": EmbeddingPricing("text-embedding-3-large", tokens_per_1m=0.13),
}


def calculate_chat_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    provider: str = "openai",
) -> float:
    """
    Calculate USD cost of a chat completion.

    Args:
        model_name: Model identifier (e.g., "gpt-4o", "gpt-4-turbo")
        input_tokens: Number of tokens in the prompt
        output_tokens: Number of tokens in the response
        provider: "openai" or "azure_openai"

    Returns:
        Cost in USD (float), rounded to 6 decimal places.
        Returns 0.0 if the model is not in the pricing table.
    """
    models = AZURE_OPENAI_MODELS if provider == "azure_openai" else OPENAI_MODELS
    pricing = models.get(model_name)

    if not pricing:
        # Model not found in pricing table; return 0 to avoid breaking on new/unknown models
        # In production, log a warning here
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing.input_tokens_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_tokens_per_1m
    total_cost = input_cost + output_cost

    return round(total_cost, 6)


def calculate_embedding_cost(
    num_tokens: int,
    model_name: str = "text-embedding-3-small",
    provider: str = "openai",
) -> float:
    """
    Calculate USD cost of embedding API call(s).

    Args:
        num_tokens: Total tokens embedded (sum across all input strings)
        model_name: Embedding model (e.g., "text-embedding-3-small")
        provider: "openai" or "azure_openai"

    Returns:
        Cost in USD (float), rounded to 6 decimal places.
        Returns 0.0 if the model is not in the pricing table.
    """
    models = AZURE_EMBEDDING_MODELS if provider == "azure_openai" else EMBEDDING_MODELS
    pricing = models.get(model_name)

    if not pricing:
        return 0.0

    cost = (num_tokens / 1_000_000) * pricing.tokens_per_1m
    return round(cost, 6)


def get_model_pricing(
    model_name: str,
    provider: str = "openai",
) -> Optional[ModelPricing]:
    """Get pricing information for a specific model."""
    models = AZURE_OPENAI_MODELS if provider == "azure_openai" else OPENAI_MODELS
    return models.get(model_name)


def get_embedding_pricing(
    model_name: str,
    provider: str = "openai",
) -> Optional[EmbeddingPricing]:
    """Get pricing information for a specific embedding model."""
    models = AZURE_EMBEDDING_MODELS if provider == "azure_openai" else EMBEDDING_MODELS
    return models.get(model_name)
