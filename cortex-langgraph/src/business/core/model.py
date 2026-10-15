"""Chat model factory — LangChain edition.

In cortex-core this file held a hand-written `BaseLLM` interface with two
implementations (`OpenAIModel` wrapping the raw OpenAI client, `LocalHFModel`
wrapping transformers + torch directly), each with its own `generate()`.

LangChain already defines that interface: `BaseChatModel`. Every provider
package ships an implementation of it, so the three branches below just pick
one. Everything downstream (the RAG graph, the agent graph) only ever sees a
`BaseChatModel` and calls `.invoke()` / `.bind_tools()` on it — it never
knows which provider is behind it.

What LangChain gives us for free compared to cortex-core:
- One call shape for every provider: `llm.invoke(messages) -> AIMessage`.
- Token usage on every response (`AIMessage.usage_metadata`), which cortex-core
  never surfaced — that is what finally makes cost tracking work end-to-end.
- Tool calling (`.bind_tools()`) on the same object, so the agent no longer
  needs a separate raw OpenAI client.
- Built-in retries (`max_retries`).
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from src.utils.config import model_settings

load_dotenv()


# ── Provider selection ────────────────────────────────────────────────────
#
# LLM_PROVIDER (env var) picks the chat model implementation. On-prem/local
# deployments don't need to set anything — the default is today's behavior.
#
#   openai        (default) — ChatOpenAI, OpenAI's cloud API. Model name from
#                   config.yml → models.<role>.name.
#   huggingface   — ChatHuggingFace over a local transformers pipeline;
#                   no network calls at generation time. Model: HF_MODEL_NAME.
#   azure_openai  — AzureChatOpenAI. Requires AZURE_OPENAI_ENDPOINT,
#                   AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
#                   (see .env for the full list).

def resolve_provider(provider: Optional[str] = None) -> str:
    """Normalise the provider name the same way everywhere (factory, agent, cost)."""
    return (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()


def create_llm(
    provider: Optional[str] = None,
    *,
    role: str = "chat",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """Factory for the chat model, selected by LLM_PROVIDER or the `provider` arg.

    role: which entry of config.yml's `models:` to use — "chat" (the agent) or
          "rag" (RAG answers). The model name and sampling settings come from
          there, so no caller names a model in code.

    Explicit arguments override the config (tests, evals, one-off scripts).
    Sampling settings the config doesn't set are left to the provider's
    defaults, as cortex-core's raw `chat.completions.create()` call did.
    """
    provider = resolve_provider(provider)
    settings = model_settings(role)
    sampling = {}
    temperature = temperature if temperature is not None else settings.get("temperature")
    max_tokens = max_tokens if max_tokens is not None else settings.get("max_tokens")
    if temperature is not None:
        sampling["temperature"] = temperature
    if max_tokens is not None:
        sampling["max_tokens"] = max_tokens

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY must be set for LLM_PROVIDER=openai")
        return ChatOpenAI(
            model=model_name or settings["name"],
            api_key=resolved_key,
            max_retries=5,
            **sampling,
        )

    if provider == "huggingface":
        # Imported lazily: transformers + torch are heavy, and only this
        # provider needs them at generation time.
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        # config.yml's names are OpenAI models, so the local model comes from
        # .env — there is no sensible default to fall back to.
        model_id = model_name or os.getenv("HF_MODEL_NAME")
        if not model_id:
            raise RuntimeError(
                "LLM_PROVIDER=huggingface requires HF_MODEL_NAME (a Hugging Face "
                "model id, e.g. 'mistralai/Mistral-7B-Instruct-v0.2')."
            )
        pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={
                "return_full_text": False,
                **({"max_new_tokens": max_tokens} if max_tokens is not None else {}),
            },
        )
        return ChatHuggingFace(llm=pipeline)

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        deployment = model_name or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        if not deployment:
            raise RuntimeError(
                "LLM_PROVIDER=azure_openai requires AZURE_OPENAI_CHAT_DEPLOYMENT_NAME "
                "(the deployment name you gave the chat model in Azure — not the "
                "underlying model name)."
            )
        endpoint, resolved_key, api_version = azure_openai_settings(api_key)
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=resolved_key,
            api_version=api_version,
            azure_deployment=deployment,
            max_retries=5,
            **sampling,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Supported: openai, huggingface, azure_openai."
    )


def azure_openai_settings(api_key: Optional[str] = None) -> tuple[str, str, str]:
    """Shared Azure OpenAI connection settings — used by both create_llm() and
    create_embedder(), so the two can never disagree about which endpoint,
    key and API version they talk to."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    resolved_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    if not endpoint or not resolved_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set to use "
            "an azure_openai provider."
        )
    return endpoint, resolved_key, api_version


def model_label(llm: BaseChatModel) -> str:
    """Best-effort model name for metrics and the API response.

    ChatOpenAI exposes `model_name`, AzureChatOpenAI `deployment_name`, and
    ChatHuggingFace `model_id`; fall back to the class name for anything else
    (including the fake models used in tests).
    """
    for attr in ("deployment_name", "model_name", "model_id", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(llm).__name__
