#Here is the openai LLM model implementation

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
import torch
from .prompt_builder import PromptBuilder
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
from typing import Optional
load_dotenv()

# Base LLM Interface
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, question: str, context: list[str]) -> str:
        pass
# abstractmethod used when you have 2 or more models that you will decide to use which one to use at runtime

#HuggingFace Local LLM Implementation
class LocalHFModel(BaseLLM):
    """
    Wrapper for openai LLMs to generate responses based on prompts.
    Local Hugging Face Model Implementation
    suitable for on-prem or offline inference
    """
    def __init__(self, model_name: str, system_prompt: str, max_input_tokens: int = 2048, max_output_tokens: int = 512):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.prompt_builder = PromptBuilder(system_prompt)
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        """
            Simple meaning:
            If the tokenizer does not define a padding token,
            the end-of-sequence (EOS) token is used as the padding token.

            Explanation:
            The pad_token is required to make all sequences in a batch the same length.

            Example:
                "Hello"
                "Hello how are you"

            Converted to tensors:
                [Hello, <PAD>, <PAD>]
                [Hello, how, are, you]

            Without a pad_token, batch processing will fail and raise runtime errors.

            The eos_token (End Of Sequence) indicates the end of a sentence.
            Many autoregressive models (e.g., GPT-style):
                - Are not designed for batch padding
                - Do not define a pad_token (pad_token = None)
                - Only provide an eos_token

            Using EOS as PAD:
                - Prevents crashes and runtime errors
                - Does not negatively affect attention mechanisms
                - Is a common and safe practice during inference
        """
    def generate(self, question: str, context: list[str]) -> str:
        """
        Generates a response from the LLM based on the provided question and context.
        
        Args:
            question (str): The user's question or input.
            context (list): Relevant context passages retrieved from the knowledge base.
            max_length (int): Maximum length of the generated response.
        
        Returns:
            str: The generated response from the LLM.
        """
        prompt = self.prompt_builder.build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,max_length=self.max_input_tokens)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=self.max_input_tokens, pad_token_id=self.tokenizer.eos_token_id)
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the part after the prompt
        return generated_text[len(prompt):].strip()

#OPENAI LLM Implementation  

class OpenAIModel(BaseLLM):
    def __init__(self, client: OpenAI, model_name: str, system_prompt:str, temperature: float = 0.7, max_tokens: int = 512):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_builder = PromptBuilder(system_prompt)
    def generate(self, question: str, context: list[str]) -> str:
        """
        Generates a response from the OpenAI LLM based on the provided question and context.
        Args:
            question (str): The user's question or input.
            context (list): Relevant context passages retrieved from the knowledge base.
        Returns:
            str: The generated response from the LLM.
        """ 
        messages = self.prompt_builder.build_messages(question, context)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()


# ── Provider selection ────────────────────────────────────────────────────
#
# LLM_PROVIDER (env var) picks the chat model implementation. On-prem/local
# deployments don't need to set anything — the default is today's behavior.
#
#   openai        (default) — OpenAI's cloud API. Current on-prem/as-is setup.
#   huggingface   — fully local inference, no network calls at generation time.
#   azure_openai  — Azure OpenAI Service. Requires AZURE_OPENAI_ENDPOINT,
#                   AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
#                   (see .env for the full list).

def build_azure_openai_client(api_key: Optional[str] = None) -> AzureOpenAI:
    """Shared AzureOpenAI client builder — used by both create_llm() and any
    caller that needs a raw client (e.g. the agentic tool-calling loop)."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    resolved_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    if not endpoint or not resolved_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set to use "
            "an azure_openai provider."
        )
    return AzureOpenAI(azure_endpoint=endpoint, api_key=resolved_key, api_version=api_version)


def create_llm(
    provider: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
    system_prompt: str = "",
    api_key: Optional[str] = None,
) -> BaseLLM:
    """Factory for the chat LLM, selected by LLM_PROVIDER or the `provider` arg."""
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()

    if provider == "openai":
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY must be set for LLM_PROVIDER=openai")
        return OpenAIModel(
            client=OpenAI(api_key=resolved_key),
            model_name=model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
            system_prompt=system_prompt,
        )

    if provider == "huggingface":
        return LocalHFModel(
            model_name=model_name or os.getenv("HF_MODEL_NAME", "mistral-7b-instruct-v0.1"),
            system_prompt=system_prompt,
        )

    if provider == "azure_openai":
        deployment = model_name or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        if not deployment:
            raise RuntimeError(
                "LLM_PROVIDER=azure_openai requires AZURE_OPENAI_CHAT_DEPLOYMENT_NAME "
                "(the deployment name you gave the chat model in Azure — not the "
                "underlying model name)."
            )
        return OpenAIModel(
            client=build_azure_openai_client(api_key),
            model_name=deployment,
            system_prompt=system_prompt,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Supported: openai, huggingface, azure_openai."
    )
