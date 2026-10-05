"""Tests for prompt_builder — the RAG ChatPromptTemplate and the agentic prompt.

Prompts are the least-tested and most behaviour-defining part of an LLM
app: a dropped instruction here doesn't raise, it just makes the model
worse in a way no other test notices. These assert on the invariants the
rest of the system depends on — the anti-hallucination rules, the
grounding of "today", and the fact that recalled context actually reaches
the model.
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.business.core.prompt_builder import (
    build_agentic_system_prompt,
    build_rag_prompt,
    format_context,
)


def render(system_prompt, question, chunks):
    """Render the RAG template exactly as the generate node does."""
    docs = [Document(page_content=c) for c in chunks]
    return build_rag_prompt(system_prompt).invoke(
        {"question": question, "context": format_context(docs)}
    ).to_messages()


# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------
class TestBuildRagPrompt:
    def test_is_a_chat_prompt_template(self):
        """A Runnable, so the RAG graph can compose it: prompt | llm | parser."""
        assert isinstance(build_rag_prompt("sys"), ChatPromptTemplate)

    def test_only_question_and_context_are_left_to_fill(self):
        """The system prompt is bound up front with .partial()."""
        assert set(build_rag_prompt("sys").input_variables) == {"question", "context"}

    def test_custom_system_prompt_is_used(self):
        assert render("You are a pirate.", "q", [])[0].content == "You are a pirate."

    def test_falls_back_to_config_when_none(self):
        assert render(None, "q", [])[0].content

    def test_default_system_role_is_the_configured_one(self):
        """Regression: a duplicate `llm_config:` key in config.yml used to wipe
        the configured role, so the RAG prompt silently fell back to the
        generic "You are a helpful assistant."."""
        from src.utils.config import load_config

        configured = load_config()["prompts"]["rag_system_role"]
        assert render(None, "q", [])[0].content == configured
        assert configured != "You are a helpful assistant."

    def test_changing_the_config_changes_the_system_role(self, override_config):
        override_config({"prompts": {"rag_system_role": "You are a terse librarian."}})
        assert render(None, "q", [])[0].content == "You are a terse librarian."

    def test_empty_string_falls_back_to_config(self):
        assert render("", "q", [])[0].content


class TestRenderedRagMessages:
    def test_returns_system_then_human(self):
        messages = render("sys", "q", ["c"])
        assert [type(m) for m in messages] == [SystemMessage, HumanMessage]

    def test_human_message_carries_question_and_context(self):
        human = render("sys", "What is X?", ["relevant chunk"])[1].content
        assert "What is X?" in human and "relevant chunk" in human

    def test_includes_every_context_chunk(self):
        human = render("sys", "q", ["alpha", "beta", "gamma"])[1].content
        assert "alpha" in human and "beta" in human and "gamma" in human

    def test_context_chunks_are_numbered_from_one(self):
        """Numbering lets the model cite which chunk it used."""
        human = render("sys", "q", ["a", "b"])[1].content
        assert "[Context 1]" in human and "[Context 2]" in human

    def test_carries_the_grounding_rules(self):
        """These rules ARE the hallucination guard for the RAG path.

        The re-ranker's min_score gate is the first firewall; this is the
        second. Losing them silently converts the RAG endpoint into an
        ungrounded chat endpoint.
        """
        human = render("sys", "q", ["ctx"])[1].content
        assert "only on the provided context" in human
        assert "I don't know." in human
        assert "Do not invent information." in human

    def test_empty_context_still_produces_a_prompt_with_the_rules(self):
        """With no context the "say I don't know" instruction is what stops
        the model answering from memory."""
        assert "I don't know." in render("sys", "q", [])[1].content

    def test_braces_in_chunks_are_not_treated_as_template_variables(self):
        """PDF text is full of `{` / `}` (code, JSON, formulas). Context is a
        template VALUE, so it must be inserted verbatim, never parsed."""
        human = render("sys", "q", ['{"key": "{value}"}'])[1].content
        assert '{"key": "{value}"}' in human


class TestFormatContext:
    def test_numbers_chunks_from_one_in_order(self):
        docs = [Document(page_content="a"), Document(page_content="b")]
        assert format_context(docs) == "[Context 1]: a\n\n[Context 2]: b"

    def test_empty_context_is_empty_string(self):
        assert format_context([]) == ""


# ---------------------------------------------------------------------------
# Agentic system prompt
# ---------------------------------------------------------------------------
class TestAgenticSystemPromptUserInfo:
    def test_renders_user_info_as_key_value_lines(self):
        prompt = build_agentic_system_prompt({"name": "Amir", "city": "Toronto"})
        assert "name: Amir" in prompt and "city: Toronto" in prompt

    def test_empty_user_info_gets_a_placeholder(self):
        assert "(no user info)" in build_agentic_system_prompt({})

    def test_none_user_info_does_not_raise(self):
        assert "(no user info)" in build_agentic_system_prompt(None)


class TestAgenticSystemPromptMemory:
    def test_recalled_snippets_are_included_and_numbered(self):
        results = [{"text": "user is a data engineer"}, {"text": "user lives in Toronto"}]
        prompt = build_agentic_system_prompt({}, vector_results=results)
        assert "[1] user is a data engineer" in prompt
        assert "[2] user lives in Toronto" in prompt

    def test_no_recall_gets_a_placeholder(self):
        assert "(no relevant past conversations found)" in build_agentic_system_prompt({})

    def test_empty_recall_list_gets_a_placeholder(self):
        assert "(no relevant past conversations found)" in \
            build_agentic_system_prompt({}, vector_results=[])

    def test_summary_is_included(self):
        prompt = build_agentic_system_prompt({}, chat_summary="User: hi\nAssistant: hello")
        assert "User: hi" in prompt

    def test_summary_is_stripped(self):
        assert "\n\n\nUser: hi" not in build_agentic_system_prompt({}, chat_summary="\n\n  User: hi  \n")

    def test_no_summary_gets_a_placeholder(self):
        assert "(no summary yet)" in build_agentic_system_prompt({})


class TestAgenticSystemPromptDateGrounding:
    """Without a stated date the model uses its training cutoff as "now".

    That made web_search actively harmful: it appended a stale year to
    queries, then dismissed the fresh results it got back as implausibly
    future-dated. This is load-bearing, not decoration.
    """

    def test_states_todays_date(self):
        prompt = build_agentic_system_prompt({})
        assert date.today().strftime("%d %B %Y").lstrip("0") in prompt.replace(" 0", " ")

    def test_includes_the_current_year(self):
        assert str(date.today().year) in build_agentic_system_prompt({})

    def test_tells_the_model_its_training_data_is_older(self):
        assert "older than this" in build_agentic_system_prompt({})


class TestAgenticSystemPromptToolInstructions:
    def test_instructs_recall_before_answering(self):
        assert "search_vector_db" in build_agentic_system_prompt({})

    def test_instructs_web_search_for_current_information(self):
        assert "web_search" in build_agentic_system_prompt({})

    def test_forbids_adding_a_year_to_search_queries(self):
        """The model biasing queries toward its training period is exactly
        what made the search results useless before."""
        assert "never add a year" in build_agentic_system_prompt({})

    def test_tells_the_model_to_trust_search_over_training_data(self):
        prompt = build_agentic_system_prompt({})
        assert "more current than your training data" in prompt

    def test_instructs_admitting_uncertainty(self):
        """The anti-hallucination instruction on the chat path."""
        prompt = build_agentic_system_prompt({})
        assert "uncertain" in prompt and "inventing information" in prompt


class TestAgenticSystemPromptStructure:
    def test_all_sections_are_present(self):
        prompt = build_agentic_system_prompt({"name": "A"}, [{"text": "m"}], "summary")
        for heading in ("User profile:", "Conversation summary so far:",
                        "Relevant past conversations", "Instructions:"):
            assert heading in prompt

    def test_returns_a_non_trivial_string(self):
        assert len(build_agentic_system_prompt({})) > 200


class TestAgenticSystemPromptTemplateSafety:
    def test_braces_in_user_data_are_inserted_verbatim(self):
        """The prompt is now a PromptTemplate. Recalled conversations and
        profile values are template VALUES — braces inside them must come
        through untouched, not be parsed as variables."""
        prompt = build_agentic_system_prompt(
            {"bio": "writes {curly} code"},
            vector_results=[{"text": 'user: parse {"a": 1}', "metadata": {}, "score": 0.1}],
            chat_summary="summary with {braces}",
        )
        assert "writes {curly} code" in prompt
        assert 'parse {"a": 1}' in prompt
        assert "summary with {braces}" in prompt
