"""Retrieve → rerank → prompt → generate."""

from __future__ import annotations

from typing import List, Tuple

from dotenv import load_dotenv

from ..core.embedding import create_embedder
from .vector_store import create_vector_store
from .re_ranker.interface import RetrievedChunk, ReRankedChunk
from .re_ranker.re_ranker import ReRanker
from .re_ranker.config import ReRankerConfig
from .re_ranker.cross_encoder import CrossEncoderReRanker
from .re_ranker.orchestrator import select_context
from ..core.prompt_builder import PromptBuilder
from ..core.model import create_llm


class RAGPipeline:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "pdf_chunks",
        reranker_config: ReRankerConfig | None = None,
        system_prompt: str | None = None,
        model_name: str = "gpt-4o-mini",
    ):
        load_dotenv()

        # Each factory reads its own *_PROVIDER env var (defaulting to
        # today's OpenAI-based behavior) and only requires the credentials
        # its selected provider actually needs.
        self.embedder = create_embedder()
        self.vector_store = create_vector_store(persist_dir=persist_dir, collection_name=collection_name)

        scorer = CrossEncoderReRanker(reranker_config or ReRankerConfig())
        self.reranker = ReRanker(scorer=scorer, config=reranker_config or ReRankerConfig())

        self.reranker_config = reranker_config or ReRankerConfig()
        self.prompt_builder = PromptBuilder(system_prompt)
        self.llm = create_llm(model_name=model_name, system_prompt=system_prompt)

    def _retrieve(self, query: str, top_k: int = 30) -> List[RetrievedChunk]:
        q_emb = self.embedder.embed_query(query)
        result = self.vector_store.query(q_emb, top_k)
        retrieved: List[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc,
                    metadata=meta or {},
                    vector_score=float(dist) if dist is not None else 0.0,
                )
            )
        return retrieved
# The answer function is the orchestration of the RAG pipeline not just the retrieval step.
    def answer(self, question: str) -> Tuple[str, List[ReRankedChunk]]:
        # Step 1: Retrieve relevant chunks from vector store
        retrieved_chunks = self._retrieve(question, top_k=self.reranker_config.top_k_input)

        # Step 2: Re-rank and select context (orchestrator handles reranking internally)
        selected_context, confidence = select_context(
            query=question,
            retrieved_chunks=retrieved_chunks,
            reranker=self.reranker,
            policy="hybrid",
        )

        # Step 3: Generate answer using LLM
        answer_text = self.llm.generate(question, [chunk.text for chunk in selected_context])

        return answer_text, selected_context
