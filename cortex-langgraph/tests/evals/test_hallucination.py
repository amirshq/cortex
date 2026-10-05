"""Hallucination / groundedness evaluation (marked `eval`).

The failure mode this suite exists for: the system produces a fluent,
confident, well-formatted answer that is not supported by anything in the
corpus. Nothing else in the test suite can catch that — a hallucinated
answer has the same type, the same shape, and the same HTTP status as a
correct one.

The corpus is fictional by construction, so any correct-sounding fact the
model produces about Veldrin Corp that is NOT in the retrieved context is
necessarily fabricated. That is what makes these assertions decidable.
"""

from __future__ import annotations

import os
from typing import List

import pytest

pytestmark = pytest.mark.eval


# --- Thresholds -------------------------------------------------------------
# Fraction of answerable questions whose answer contains the correct fact.
MIN_GROUNDED_ACCURACY = 0.75
# Fraction of unanswerable questions the system declines rather than invents.
# Set high on purpose: confidently inventing a warranty term or a revenue
# figure is worse than being useless, because a user cannot tell.
MIN_REFUSAL_RATE = 0.75
# LLM-as-judge groundedness over the answerable set.
MIN_JUDGE_GROUNDEDNESS = 0.75

REFUSAL_MARKERS = (
    "i don't know", "i do not know", "not in the context", "no information",
    "does not contain", "doesn't contain", "not provided", "not mentioned",
    "not specified", "cannot determine", "can't determine", "unable to find",
    "no mention", "not stated", "not available",
)


def rag_generate(question: str, context: List[str]) -> str:
    """The RAG generate step on its own: the production prompt | model | parser
    chain, with the same model and sampling settings as RAGPipeline
    (config.yml → models.rag)."""
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser

    from src.business.core.model import create_llm
    from src.business.core.prompt_builder import build_rag_prompt, format_context

    chain = build_rag_prompt() | create_llm(role="rag") | StrOutputParser()
    return chain.invoke({"question": question,
                         "context": format_context([Document(page_content=c) for c in context])})


def looks_like_a_refusal(answer: str) -> bool:
    return any(marker in answer.lower() for marker in REFUSAL_MARKERS)


class TestGroundedAnswers:
    """Questions the corpus DOES answer must be answered correctly."""

    def test_answers_contain_the_corpus_fact(self, rag_pipeline, golden_set):
        cases = golden_set["grounded_cases"]
        hits = 0
        print()
        for case in cases:
            answer, _, _ = rag_pipeline.answer(case["question"])
            grounded = any(token.lower() in answer.lower() for token in case["must_contain_any"])
            hits += grounded
            print(f"  [{'ok  ' if grounded else 'MISS'}] {case['question'][:50]!r}")
            if not grounded:
                print(f"         expected one of {case['must_contain_any']}")
                print(f"         got: {answer[:160]!r}")

        accuracy = hits / len(cases)
        print(f"\n  grounded accuracy = {accuracy:.3f}  ({hits}/{len(cases)})")
        assert accuracy >= MIN_GROUNDED_ACCURACY

    def test_answerable_questions_are_not_wrongly_refused(self, rag_pipeline, golden_set):
        """Over-refusal is the opposite failure: a system that says
        "I don't know" to everything passes the hallucination tests and is
        still worthless."""
        refused = [
            c["question"] for c in golden_set["grounded_cases"]
            if looks_like_a_refusal(rag_pipeline.answer(c["question"])[0])
        ]
        print(f"\n  wrongly refused: {len(refused)}/{len(golden_set['grounded_cases'])}")
        assert not refused, f"refused answerable questions: {refused}"

    def test_sources_are_returned_with_every_answer(self, rag_pipeline, golden_set):
        """An answer without sources cannot be verified by the user, which
        is the only defence left once the model is fluent."""
        for case in golden_set["grounded_cases"]:
            _, chunks, _ = rag_pipeline.answer(case["question"])
            assert chunks, f"no sources returned for {case['question']!r}"


class TestRefusalOnUnanswerable:
    """Questions the corpus does NOT answer must be declined."""

    def test_refuses_rather_than_inventing(self, rag_pipeline, golden_set):
        cases = golden_set["unanswerable_cases"]
        refusals = 0
        print()
        for case in cases:
            answer, _, _ = rag_pipeline.answer(case["question"])
            refused = looks_like_a_refusal(answer)
            refusals += refused
            print(f"  [{'ok  ' if refused else 'HALLUCINATED'}] {case['question'][:52]!r}")
            if not refused:
                print(f"         ({case['why']})")
                print(f"         answered: {answer[:160]!r}")

        rate = refusals / len(cases)
        print(f"\n  refusal rate = {rate:.3f}  ({refusals}/{len(cases)})")
        assert rate >= MIN_REFUSAL_RATE, (
            f"refusal rate {rate:.3f} < {MIN_REFUSAL_RATE}. The system is "
            f"inventing facts about a company that does not exist — in "
            f"production that is a confidently wrong answer with sources "
            f"attached."
        )

    def test_does_not_invent_a_nonexistent_product(self, rag_pipeline):
        """The corpus has a Kestrel-7. There is no Kestrel-9. A model that
        answers about one is pattern-matching, not retrieving."""
        answer, _, _ = rag_pipeline.answer("What is the battery life of the Kestrel-9?")
        print(f"\n  answer: {answer[:200]!r}")
        assert looks_like_a_refusal(answer) or "kestrel-7" in answer.lower(), (
            "invented specifications for a product not in the corpus"
        )

    def test_does_not_extrapolate_beyond_the_corpus_years(self, rag_pipeline):
        """Financials stop at FY2024. A fluent trend-extrapolation to 2026
        reads exactly like a retrieved fact."""
        answer, _, _ = rag_pipeline.answer("What was Veldrin's revenue in fiscal 2026?")
        print(f"\n  answer: {answer[:200]!r}")
        assert looks_like_a_refusal(answer) or "2024" in answer, (
            "extrapolated a revenue figure for a year absent from the corpus"
        )


class TestEmptyContextBehaviour:
    """The fail-closed path: what happens when retrieval returns nothing."""

    def test_no_context_produces_a_refusal_not_an_answer(self, require_openai_key):
        """With zero context the model must fall back on the prompt's
        "I don't know" rule rather than its own training data."""
        answer = rag_generate("What is Veldrin Corp's employee count?", [])
        print(f"\n  answer with no context: {answer[:200]!r}")
        assert looks_like_a_refusal(answer), (
            "answered from training data with no retrieved context — the "
            "grounding rules in PromptBuilder are not taking effect"
        )

    def test_irrelevant_context_is_not_forced_into_an_answer(self, require_openai_key):
        """Given only the off-topic distractor, the model must not stretch
        it into an answer about the company."""
        answer = rag_generate(
            "How many people does Veldrin Corp employ?",
            ["Relative humidity is the ratio of the partial pressure of water "
             "vapour to the equilibrium vapour pressure at a given temperature."],
        )
        print(f"\n  answer from irrelevant context: {answer[:200]!r}")
        assert looks_like_a_refusal(answer)


class TestLLMAsJudge:
    """A second model checks whether each answer is entailed by its sources.

    Keyword matching catches an answer that omits the right number; it
    cannot catch one that includes the right number surrounded by invented
    detail. A judge can.
    """

    JUDGE_PROMPT = (
        "You are a strict grader. You will be given CONTEXT and an ANSWER.\n"
        "Reply with exactly one word: GROUNDED if every factual claim in the "
        "ANSWER is directly supported by the CONTEXT, or UNGROUNDED if the "
        "ANSWER contains any claim not present in the CONTEXT. A refusal to "
        "answer counts as GROUNDED. Reply with one word only."
    )

    def _judge(self, context: List[str], answer: str) -> bool:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": self.JUDGE_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{chr(10).join(context)}\n\nANSWER:\n{answer}"},
            ],
        )
        return "UNGROUNDED" not in response.choices[0].message.content.upper()

    def test_answers_are_entailed_by_their_own_sources(self, rag_pipeline, golden_set):
        cases = golden_set["grounded_cases"]
        grounded = 0
        print()
        for case in cases:
            answer, chunks, _ = rag_pipeline.answer(case["question"])
            verdict = self._judge([c.page_content for c in chunks], answer)
            grounded += verdict
            print(f"  [{'GROUNDED  ' if verdict else 'UNGROUNDED'}] {case['question'][:48]!r}")
            if not verdict:
                print(f"         {answer[:160]!r}")

        rate = grounded / len(cases)
        print(f"\n  judged groundedness = {rate:.3f}  ({grounded}/{len(cases)})")
        assert rate >= MIN_JUDGE_GROUNDEDNESS

    def test_the_judge_itself_detects_a_planted_hallucination(self):
        """Calibration. A judge that says GROUNDED to everything would make
        the test above pass unconditionally and mean nothing.
        """
        context = ["Veldrin Corp employs 412 people as of the 2024 annual report."]
        assert self._judge(context, "Veldrin employs 412 people."), \
            "judge rejected a correct answer — it is too strict to trust"
        assert not self._judge(
            context,
            "Veldrin employs 412 people and was acquired by Siemens in 2025 "
            "for 1.2 billion euros.",
        ), "judge accepted a fabricated acquisition — it is not detecting anything"
