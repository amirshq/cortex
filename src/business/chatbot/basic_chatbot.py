"""Basic conversational chatbot using OpenAI."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.config import load_config


class BasicChatbot:
    """
    Simple chatbot that sends user messages to an LLM and returns responses.
    No RAG, No Memory, no tools—just direct conversation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY must be set in environment or passed to __init__")

        config = load_config()
        llm_config = config.get("llm_config") or {}

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name or config.get("llm_model_name") or llm_config.get("chat_model") or "gpt-3.5-turbo"
        self.system_prompt = system_prompt or llm_config.get("llm_system_role") or "You are a helpful assistant."
        self.temperature = temperature if temperature is not None else config.get("temperature", 0.7)
        self.max_tokens = max_tokens or config.get("max_token") or 1500

    def chat(self, message: str) -> str:
        """Handle a conversation of the user and the conversatio history"""
        self.previous_summary = self.


def main():
    """Run a simple REPL chat loop."""
    chatbot = BasicChatbot()
    print("Basic Chatbot — type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        reply = chatbot.chat(user_input)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
