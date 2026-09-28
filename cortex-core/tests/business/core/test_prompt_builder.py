"""Tests for prompt_builder — both the RAG PromptBuilder and the agentic one.

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

from src.business.core.prompt_builder import PromptBuilder, build_agentic_system_prompt


# ---------------------------------------------------------------------------
# RAG prompt builder
# ---------------------------------------------------------------------------
class TestPromptBuilderInit:
    def test_custom_system_prompt_is_used(self):
        assert PromptBuilder("You are a pirate.").system_prompt == "You are a pirate."

    def test_falls_back_to_config_when_none(self):
        assert PromptBuilder().system_prompt

    def test_empty_string_falls_back_to_config(self):
        assert PromptBuilder("").system_prompt


class TestBuildPromptText:
    def test_includes_the_question(self):
        assert "What is X?" in PromptBuilder("sys").build_prompt_text("What is X?", ["ctx"])

    def test_includes_every_context_chunk(self):
        prompt = PromptBuilder("sys").build_prompt_text("q", ["alpha", "beta", "gamma"])
        assert "alpha" in prompt and "beta" in prompt and "gamma" in prompt

    def test_context_chunks_are_numbered_from_one(self):
        """Numbering lets the model cite which chunk it used."""
        prompt = PromptBuilder("sys").build_prompt_text("q", ["a", "b"])
        assert "[Context 1]" in prompt and "[Context 2]" in prompt

    def test_includes_the_system_prompt(self):
        assert "CUSTOM ROLE" in PromptBuilder("CUSTOM ROLE").build_prompt_text("q", [])

    def test_carries_the_grounding_rules(self):
        """These four rules ARE the hallucination guard for the RAG path.

        The re-ranker's min_score gate is the first firewall; this is the
        second. Losing them silently converts the RAG endpoint into an
        ungrounded chat endpoint.
        """
        prompt = PromptBuilder("sys").build_prompt_text("q", ["ctx"])
        assert "only on the provided context" in prompt
        assert "I don't know." in prompt
        assert "Do not invent information." in prompt

    def test_empty_context_still_produces_a_prompt_with_the_rules(self):
        """The fail-closed path sends no context — the "say I don't know"
        instruction is what stops the model answering from memory."""
        prompt = PromptBuilder("sys").build_prompt_text("q", [])
        assert "I don't know." in prompt

    def test_is_stripped(self):
        prompt = PromptBuilder("sys").build_prompt_text("q", ["c"])
        assert prompt == prompt.strip()

    def test_build_prompt_alias_matches(self):
        builder = PromptBuilder("sys")
        assert builder.build_prompt("q", ["c"]) == builder.build_prompt_text("q", ["c"])


class TestBuildMessages:
    def test_returns_system_then_user(self):
        messages = PromptBuilder("sys").build_messages("q", ["c"])
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_system_message_is_the_system_prompt(self):
        assert PromptBuilder("ROLE").build_messages("q", [])[0]["content"] == "ROLE"

    def test_user_message_carries_question_and_context(self):
        user = PromptBuilder("sys").build_messages("What is X?", ["relevant chunk"])[1]["content"]
        assert "What is X?" in user and "relevant chunk" in user

    def test_user_message_carries_the_grounding_rules(self):
        user = PromptBuilder("sys").build_messages("q", ["c"])[1]["content"]
        assert "only on the provided context" in user
        assert "Do not invent information." in user

    def test_context_numbering_matches_the_text_variant(self):
        user = PromptBuilder("sys").build_messages("q", ["a", "b"])[1]["content"]
        assert "[Context 1]" in user and "[Context 2]" in user


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
