"""Prompt construction — LangChain edition.

cortex-core built prompts with f-strings inside a `PromptBuilder` class. Here
the same wording lives in LangChain prompt templates:

- `build_rag_prompt()`  → a `ChatPromptTemplate` (system + human message) that
  the RAG graph pipes straight into the chat model: `prompt | llm`.
- `build_agentic_system_prompt()` → the agent's system prompt, rendered from a
  `PromptTemplate`.

A template is a Runnable, so it composes with `|` and shows up as its own
step in LangSmith/Langfuse traces. The wording is deliberately identical to
cortex-core so the two implementations can be compared on equal footing.
"""

from datetime import date
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.utils.config import load_config


def _get_system_role() -> str:
    """The RAG system prompt, from config.yml → prompts.rag_system_role."""
    return (load_config().get("prompts") or {}).get("rag_system_role") or "You are a helpful assistant."


# ---------------------------------------------------------------------------
# RAG prompt
# ---------------------------------------------------------------------------
# Template variables: {system_prompt}, {context}, {question}. The grounding
# rules are the hallucination guardrail of the RAG path — keep them verbatim.
RAG_HUMAN_TEMPLATE = (
    "Use the following context to answer the question.\n"
    "{context}\n"
    "Question: {question}\n"
    "Rules:\n"
    "- Answer based only on the provided context.\n"
    '- If the answer is not in the context, respond with "I don\'t know."\n'
    "- Be concise and to the point.\n"
    "- Do not invent information.\n"
)


def build_rag_prompt(system_prompt: Optional[str] = None) -> ChatPromptTemplate:
    """ChatPromptTemplate for the RAG generate step.

    The system prompt is bound up front with .partial(), so the graph only has
    to supply {context} and {question} at run time.

    Args:
        system_prompt: Custom system prompt. If None, uses prompts.rag_system_role from config.yml.
    """
    template = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human", RAG_HUMAN_TEMPLATE),
    ])
    return template.partial(system_prompt=system_prompt or _get_system_role())


def format_context(docs: List[Document]) -> str:
    """Render retrieved chunks as numbered context blocks: [Context 1]: ..."""
    return "\n\n".join(f"[Context {i + 1}]: {doc.page_content}" for i, doc in enumerate(docs))


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------
AGENTIC_SYSTEM_TEMPLATE = PromptTemplate.from_template("""You are a helpful, context-aware personal assistant.

Today's date is {today}. Your own training data is older than this, so treat
anything time-sensitive as something you need to look up rather than recall.

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
- Call news_search for current events: news, "latest", "this week", "today",
  or trending topics. Set `days` to match the question ("last seven days" →
  days=7). Call web_search for general knowledge: what something is, how it
  works, comparisons, documentation. Search using the user's own terms —
  never add a year to the query yourself, as that biases the results toward
  the wrong period.
- Results from these tools are more current than your training data. When they
  conflict, trust the search results and answer from them. If the results are
  dated later than you expect, they are still correct — do not dismiss them.
- If the results don't cover a well-known topic (e.g. a popular framework or
  concept), still answer from your own knowledge, and say that the answer is
  not from the search results. Never reply only that the search found nothing.
- Search queries name the topic in full ("artificial intelligence", not "AI")
  and leave out words like "top", "hot", "latest" — those match unrelated
  articles. The time window goes in `days`, never in the query.
- When the user asks for a specific number of items ("top 5"), give exactly
  that many. Skip off-topic results, and if fewer than needed remain, search
  again with different wording (a synonym, or a narrower sub-topic) before
  answering. Only if you still have too few, give what you found and say how
  many.
- Format links in Markdown: [title](url).
- Be concise and direct. Do not repeat context back to the user verbatim.
- If you are uncertain, say so rather than inventing information.""")


def build_agentic_system_prompt(
    user_info: Dict,
    vector_results: Optional[List[Dict]] = None,
    chat_summary: Optional[str] = None,
) -> str:
    """
    Build the system prompt for the agentic chatbot.

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

    # Without this the model falls back on its training cutoff as "now": it
    # appends a stale year to web_search queries and then discounts the fresh
    # results it gets back as implausibly future-dated. Stating the real date
    # is what makes the web_search tool actually usable.
    today = date.today().strftime("%A, %d %B %Y")

    return AGENTIC_SYSTEM_TEMPLATE.format(
        today=today,
        user_block=user_block,
        summary_block=summary_block,
        snippets=snippets,
    )
