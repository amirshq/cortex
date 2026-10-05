"""Tests for RAGPipeline — the retrieve → rerank → (gate) → generate graph.

Hermetic: a real Chroma store in tmp_path (FakeEmbedder), a scripted
cross-encoder, and a FakeChatModel. Covers what cortex-core spread over
test_retrieval.py and test_orchestrator.py: the vector-store mapping, what
reaches the LLM, and the three policy branches (now conditional edges).
"""

from __future__ import annotations

from typing import List

import pytest
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.business.rag.re_ranker.config import ReRankerConfig
from src.business.rag.retrieval import REFUSAL, RAGPipeline
from src.business.rag.vector_store import create_vector_store
from tests.conftest import FakeChatModel, reply


class ScriptedScorer(BaseCrossEncoder):
    """Scores each chunk by looking its text up in a dict (default 0.0)."""

    def __init__(self, scores=None, default: float = 0.0):
        self.scores = scores or {}
        self.default = default
        self.calls: List[list] = []

    def score(self, text_pairs):
        self.calls.append(list(text_pairs))
        return [self.scores.get(doc, self.default) for _, doc in text_pairs]


@pytest.fixture
def store(tmp_path, fake_embedder):
    vs = create_vector_store(str(tmp_path / "rag"), embedding=fake_embedder, provider="chroma")
    texts = [f"chunk {i}" for i in range(12)]
    vs.add_documents(
        [Document(page_content=t, metadata={"source_id": "doc.pdf", "chunk_start": i * 10})
         for i, t in enumerate(texts)],
        ids=[f"id-{i}" for i in range(12)],
    )
    return vs


def build(store, scorer, *, answer="the answer", policy="hybrid", config=None, system_prompt="SYS"):
    llm = FakeChatModel(responses=[reply(answer)])
    pipeline = RAGPipeline(
        persist_dir="unused",
        vector_store=store,
        scorer=scorer,
        llm=llm,
        policy=policy,
        reranker_config=config or ReRankerConfig(top_k_input=10, top_n_output=3, min_score=0.15),
        system_prompt=system_prompt,
    )
    return pipeline, llm


class TestGraphShape:
    def test_nodes(self, store):
        pipeline, _ = build(store, ScriptedScorer())
        nodes = set(pipeline.graph.get_graph().nodes)
        assert {"retrieve", "rerank", "fallback", "refuse", "generate"} <= nodes

    def test_rerank_branches_to_generate_fallback_and_refuse(self, store):
        pipeline, _ = build(store, ScriptedScorer())
        targets = {e.target for e in pipeline.graph.get_graph().edges if e.source == "rerank"}
        assert targets == {"generate", "fallback", "refuse"}


class TestRetrieveNode:
    def test_fetches_top_k_input_candidates(self, store):
        pipeline, _ = build(store, ScriptedScorer(default=1.0))
        state = pipeline.run("chunk 3")
        assert len(state["candidates"]) == 10

    def test_candidates_carry_their_vector_distance(self, store):
        pipeline, _ = build(store, ScriptedScorer(default=1.0))
        candidates = pipeline.run("chunk 3")["candidates"]
        assert all(isinstance(c.metadata["vector_score"], float) for c in candidates)
        distances = [c.metadata["vector_score"] for c in candidates]
        assert distances == sorted(distances), "best (smallest distance) first"

    def test_candidates_keep_their_id_and_metadata(self, store):
        pipeline, _ = build(store, ScriptedScorer(default=1.0))
        top = pipeline.run("chunk 3")["candidates"][0]
        assert top.id.startswith("id-")
        assert top.metadata["source_id"] == "doc.pdf"

    def test_empty_index_yields_no_candidates(self, tmp_path, fake_embedder):
        empty = create_vector_store(str(tmp_path / "empty"), embedding=fake_embedder, provider="chroma")
        pipeline, _ = build(empty, ScriptedScorer())
        state = pipeline.run("anything")
        assert state["candidates"] == []
        assert state["confidence"] == "low"


class TestHighConfidenceBranch:
    def test_only_reranked_chunks_reach_the_llm(self, store):
        scorer = ScriptedScorer({"chunk 1": 5.0, "chunk 2": 3.0}, default=-1.0)
        pipeline, llm = build(store, scorer)
        answer, context, confidence = pipeline.answer("q")

        assert confidence == "high"
        assert [d.page_content for d in context] == ["chunk 1", "chunk 2"]
        human = llm.calls[0][-1].content
        assert "chunk 1" in human and "chunk 2" in human
        assert "chunk 5" not in human

    def test_context_is_sorted_by_rerank_score(self, store):
        scorer = ScriptedScorer({"chunk 1": 1.0, "chunk 2": 9.0, "chunk 3": 4.0}, default=-1.0)
        pipeline, _ = build(store, scorer)
        _, context, _ = pipeline.answer("q")
        assert [d.page_content for d in context] == ["chunk 2", "chunk 3", "chunk 1"]

    def test_context_is_truncated_to_top_n_output(self, store):
        pipeline, _ = build(store, ScriptedScorer(default=1.0))
        _, context, _ = pipeline.answer("q")
        assert len(context) == 3

    def test_query_is_forwarded_verbatim_to_the_scorer(self, store):
        scorer = ScriptedScorer(default=1.0)
        pipeline, _ = build(store, scorer)
        pipeline.answer("What is X, exactly?")
        assert {q for q, _ in scorer.calls[0]} == {"What is X, exactly?"}

    def test_returns_the_llm_answer_stripped(self, store):
        pipeline, _ = build(store, ScriptedScorer(default=1.0), answer="  spaced answer \n")
        assert pipeline.answer("q")[0] == "spaced answer"

    def test_prompt_is_system_then_human_with_question(self, store):
        pipeline, llm = build(store, ScriptedScorer(default=1.0), system_prompt="CUSTOM ROLE")
        pipeline.answer("What is X?")
        system, human = llm.calls[0]
        assert isinstance(system, SystemMessage) and system.content == "CUSTOM ROLE"
        assert isinstance(human, HumanMessage) and "What is X?" in human.content


class TestFallbackBranch:
    """Gate rejected everything + fail_open/hybrid → raw vector order, low confidence."""

    @pytest.mark.parametrize("policy", ["hybrid", "fail_open"])
    def test_falls_back_to_vector_order(self, store, policy):
        pipeline, llm = build(store, ScriptedScorer(default=-5.0), policy=policy)
        state = pipeline.run("chunk 3")

        assert state["confidence"] == "low"
        assert state["context"] == state["candidates"][:3]
        assert len(llm.calls) == 1, "fallback still generates an answer"

    def test_fallback_chunks_have_no_rerank_score(self, store):
        pipeline, _ = build(store, ScriptedScorer(default=-5.0))
        _, context, _ = pipeline.answer("q")
        assert all("rerank_score" not in d.metadata for d in context)


class TestRefuseBranch:
    """Gate rejected everything + fail_closed → never call the LLM."""

    def test_refuses_without_calling_the_llm(self, store):
        pipeline, llm = build(store, ScriptedScorer(default=-5.0), policy="fail_closed")
        answer, context, confidence = pipeline.answer("q")

        assert answer == REFUSAL
        assert context == []
        assert confidence == "none"
        assert llm.calls == []

    def test_fail_closed_still_answers_when_the_gate_passes(self, store):
        pipeline, llm = build(store, ScriptedScorer(default=1.0), policy="fail_closed")
        assert pipeline.answer("q")[2] == "high"
        assert len(llm.calls) == 1


class TestDefaults:
    def test_default_policy_is_hybrid(self, store):
        pipeline = RAGPipeline(persist_dir="unused", vector_store=store, scorer=ScriptedScorer(),
                               llm=FakeChatModel())
        assert pipeline.policy == "hybrid"

    def test_default_rag_llm_settings_come_from_config(self, store, monkeypatch):
        from src.utils.config import model_settings

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        rag = model_settings("rag")
        pipeline = RAGPipeline(persist_dir="unused", vector_store=store, scorer=ScriptedScorer())
        assert (pipeline.llm.model_name, pipeline.llm.temperature, pipeline.llm.max_tokens) == \
            (rag["name"], rag["temperature"], rag["max_tokens"])

    def test_azure_rag_uses_the_chat_deployment(self, store, monkeypatch):
        """cortex-core sent its hardcoded "gpt-4o-mini" as the Azure deployment
        name for RAG, which only works if a deployment happens to be called that."""
        monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "my-chat-deployment")
        pipeline = RAGPipeline(persist_dir="unused", vector_store=store, scorer=ScriptedScorer())
        assert pipeline.llm.deployment_name == "my-chat-deployment"
