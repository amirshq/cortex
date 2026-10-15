"""RAG pipeline — a LangGraph StateGraph.

cortex-core's `RAGPipeline.answer()` was three function calls in a row, with
the "what if the re-ranker rejects everything?" policy buried inside
`select_context()`. Here each step is a node and that policy is a visible
branch:

    START → retrieve → rerank ──(chunks passed the gate)──────────────▶ generate → END
                          │                                               ▲
                          ├──(nothing passed, policy fail_open/hybrid)──▶ fallback
                          │
                          └──(nothing passed, policy fail_closed)───────▶ refuse → END

- retrieve  VectorStore.similarity_search_with_score — dense top-k (config.yml → rag.retrieval.top_k)
- rerank    cross-encoder score → relevance gate → sort → top-n (rag.retrieval.top_n)
          With models.reranker.name = "none" the node keeps the vector top-n
          as they are (no scoring, no gate) and confidence is "unranked".
          Chosen by the golden-set evaluation (experiment 3.4): with
          text-embedding-3-large, vector order was as good as every re-ranker
          tried, and skipping it saved ~0.6 s per question.
- fallback  gate rejected everything: use the raw vector top-n, confidence "low"
- refuse    fail_closed policy: don't call the LLM at all
- generate  ChatPromptTemplate | chat model | StrOutputParser (an LCEL chain)

Every node reads the shared state and returns only the keys it changes; LangGraph
merges them. `confidence` records which branch ran — "high" (re-ranked),
"low" (fallback) or "none" (refused) — and query_rag() turns it into metrics.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

from ..core.model import create_llm
from ..core.prompt_builder import build_rag_prompt, format_context
from .re_ranker.config import ReRankerConfig
from .re_ranker.re_ranker import rerank_documents
from .vector_store import create_vector_store
from src.utils.config import rag_settings

Policy = Literal["fail_closed", "fail_open", "hybrid"]

# What the refuse branch answers — the same wording the grounding rules tell
# the model to use when the context doesn't contain the answer.
REFUSAL = "I don't know."


class RAGState(TypedDict, total=False):
    """The data that flows through the graph. Nodes return partial updates."""
    question: str
    candidates: List[Document]   # vector-store results, best first
    context: List[Document]      # what the LLM will see
    confidence: str              # "high" | "low" | "none" | "unranked" (no re-ranker)
    answer: str


class RAGPipeline:
    """
    Retrieve → rerank → (gate) → generate, compiled as a LangGraph graph.

    Collaborators are built from the provider factories by default, and can be
    injected (tests pass fakes; evals pass a shared cross-encoder).
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "pdf_chunks",
        reranker_config: ReRankerConfig | None = None,
        system_prompt: str | None = None,
        model_name: str | None = None,
        policy: Policy | None = None,
        vector_store: Optional[VectorStore] = None,
        scorer: Optional[BaseCrossEncoder] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        load_dotenv()

        self.reranker_config = reranker_config or ReRankerConfig()
        # What to do when the gate rejects every chunk — config.yml → rag.retrieval.fallback_policy.
        self.policy = policy or rag_settings("retrieval")["fallback_policy"]
        if self.policy not in ("fail_closed", "fail_open", "hybrid"):
            raise ValueError(f"Unknown fallback_policy={self.policy!r}. Use hybrid, fail_open or fail_closed.")

        # Each factory reads its own *_PROVIDER env var (defaulting to
        # today's OpenAI-based behavior) and only requires the credentials
        # its selected provider actually needs.
        self.vector_store = vector_store or create_vector_store(
            persist_dir=persist_dir, collection_name=collection_name
        )
        # models.reranker.name = "none": no cross-encoder at all (and no torch).
        self.rerank_enabled = scorer is not None or self.reranker_config.model_name != "none"
        if scorer is None and self.rerank_enabled:
            # Imported lazily: loading the cross-encoder pulls in torch.
            from .re_ranker.cross_encoder import CrossEncoderScorer

            scorer = CrossEncoderScorer(self.reranker_config)
        self.scorer = scorer
        # role="rag": model, temperature and max_tokens come from config.yml →
        # models.rag. model_name only overrides the model (evals, experiments).
        self.llm = llm or create_llm(role="rag", model_name=model_name)

        # prompt | llm | parser — an LCEL chain, used by the generate node.
        self.generate_chain = build_rag_prompt(system_prompt) | self.llm | StrOutputParser()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ nodes
    def _retrieve(self, state: RAGState) -> RAGState:
        results = self.vector_store.similarity_search_with_score(
            state["question"], k=self.reranker_config.top_k_input
        )
        candidates = [
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={**doc.metadata, "vector_score": float(score)},
            )
            for doc, score in results
        ]
        return {"candidates": candidates}

    def _rerank(self, state: RAGState) -> RAGState:
        if not self.rerank_enabled:
            # Vector order, cut to top-n. No gate, so the fallback branch never runs.
            return {"context": state["candidates"][: self.reranker_config.top_n_output],
                    "confidence": "unranked"}
        reranked = rerank_documents(
            query=state["question"],
            documents=state["candidates"],
            scorer=self.scorer,
            config=self.reranker_config,
        )
        # An empty context tells the router below that the gate rejected everything.
        return {"context": reranked, "confidence": "high" if reranked else "none"}

    def _route_after_rerank(self, state: RAGState) -> str:
        """The policy cortex-core kept in orchestrator.select_context()."""
        if state["context"]:
            return "generate"
        if self.policy == "fail_closed":
            return "refuse"
        return "fallback"

    def _fallback(self, state: RAGState) -> RAGState:
        # Fail open: raw vector order, flagged low-confidence so it shows up in metrics.
        return {
            "context": state["candidates"][: self.reranker_config.top_n_output],
            "confidence": "low",
        }

    def _refuse(self, state: RAGState) -> RAGState:
        return {"context": [], "confidence": "none", "answer": REFUSAL}

    def _generate(self, state: RAGState) -> RAGState:
        answer = self.generate_chain.invoke({
            "question": state["question"],
            "context": format_context(state["context"]),
        })
        return {"answer": answer.strip()}

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("rerank", self._rerank)
        builder.add_node("fallback", self._fallback)
        builder.add_node("refuse", self._refuse)
        builder.add_node("generate", self._generate)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_conditional_edges(
            "rerank",
            self._route_after_rerank,
            {"generate": "generate", "fallback": "fallback", "refuse": "refuse"},
        )
        builder.add_edge("fallback", "generate")
        builder.add_edge("generate", END)
        builder.add_edge("refuse", END)
        return builder.compile()

    # -------------------------------------------------------------------- API
    def run(self, question: str) -> RAGState:
        """Run the graph and return its final state."""
        return self.graph.invoke({"question": question})

    def answer(self, question: str) -> Tuple[str, List[Document], str]:
        """Answer a question. Returns (answer, context chunks, confidence)."""
        state = self.run(question)
        return state["answer"], state.get("context", []), state.get("confidence", "none")
