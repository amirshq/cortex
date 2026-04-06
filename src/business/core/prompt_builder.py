from typing import Dict, List, Optional
import yaml
from pathlib import Path

def _load_config() -> dict:
    """Load configuration from config.yml file."""
    # core/ -> business/ -> src/ -> config/config.yml
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _get_system_role() -> str:
    """Get LLM system role from config.yml."""
    config = _load_config()
    return config.get("llm_config", {}).get("llm_system_role", "You are a helpful assistant.")

class PromptBuilder:
    """
    Responsible for constructing prompts for the AI model.
    Sent to the LLM to generate responses based on user input and context in RAG pipeline.
    """

    def __init__(self, system_prompt: str = None):
        """
        Initialize PromptBuilder.

        Args:
            system_prompt: Custom system prompt. If None, uses llm_system_role from config.yml.
        """
        # Use config value if system_prompt not provided
        self.system_prompt = system_prompt or _get_system_role()

    def build_prompt_text(self, question: str, context: List[str]) -> str:
        """
        Constructs the full prompt (plain text) for completion-style models.
        """
        context_block = "\n\n".join(f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(context))
        full_prompt = f"""
        {self.system_prompt}
        Use the following context to answer the question.
        {context_block}
        Question: {question}
        Rules:
        - Answer based only on the provided context.
        - If the answer is not in the context, respond with "I don't know."
        - Be concise and to the point.
        - Do not invent information.
        """
        return full_prompt.strip()

    def build_messages(self, question: str, context: List[str]):
        """
        Constructs chat messages for OpenAI Chat Completions API.
        """
        context_block = "\n\n".join(f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(context))
        user_msg = (
            "Use the following context to answer the question.\n"
            f"{context_block}\n"
            f"Question: {question}\n"
            "Rules:\n"
            '- Answer based only on the provided context.\n'
            '- If the answer is not in the context, respond with "I don\'t know."\n'
            "- Be concise and to the point.\n"
            "- Do not invent information.\n"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    # Backwards compatibility
    def build_prompt(self, question: str, context: List[str]) -> str:
        return self.build_prompt_text(question, context)


def build_agentic_system_prompt(
    user_info: Dict,
    vector_results: Optional[List[Dict]] = None,
    chat_summary: Optional[str] = None,
) -> str:
    """
    Build the system prompt for the agentic chatbot (V3).

    Injects user profile, an optional rolling summary, and semantically
    retrieved past-conversation snippets so the model has rich context
    without blowing up the context window with raw history.

    Args:
        user_info: Key/value pairs describing the user (name, preferences, etc.)
        vector_results: Output of LongTermMemory.recall() — list of
                        {"text": str, "metadata": dict, "score": float}
        chat_summary: Optional condensed summary of the conversation so far.
    """
    user_block = "\n".join(f"  {k}: {v}" for k, v in user_info.items()) if user_info else "  (no user info)"

    summary_block = chat_summary.strip() if chat_summary else "(no summary yet)"

    if vector_results:
        snippets = "\n\n".join(
            f"[{i + 1}] {r['text']}" for i, r in enumerate(vector_results)
        )
    else:
        snippets = "(no relevant past conversations found)"

    return f"""You are a helpful, context-aware personal assistant.

User profile:
{user_block}

Conversation summary so far:
{summary_block}

Relevant past conversations (semantic recall):
{snippets}

Instructions:
- Use the user profile and past context to personalise your responses.
- Before answering, ALWAYS call search_vector_db to look up anything the user
  may have shared previously that is relevant to their current question — such
  as their job, hobbies, interests, preferences, or any personal details.
  Do not skip this step even if the current question seems unrelated to past
  conversations; the user's background often changes what a good answer looks like.
- Be concise and direct. Do not repeat context back to the user verbatim.
- If you are uncertain, say so rather than inventing information."""
