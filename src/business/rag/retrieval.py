"""Retrieve → rerank → prompt → generate."""

from __future__ import annotations

import os
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from ..core.embedding import OpenAIEmbedder
from .vector_store import VectorStore
from .re_ranker.interface import RetrievedChunk, ReRankedChunk
from .re_ranker.re_ranker import ReRanker
from .re_ranker.config import ReRankerConfig
from .re_ranker.cross_encoder import CrossEncoderReRanker
from .re_ranker.orchestrator import select_context
from ..core.prompt_builder import PromptBuilder
from ..core.model import OpenAIModel


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
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing in environment")

        self.embedder = OpenAIEmbedder(api_key=api_key)
        self.vector_store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)

        scorer = CrossEncoderReRanker(reranker_config or ReRankerConfig())
        self.reranker = ReRanker(scorer=scorer, config=reranker_config or ReRankerConfig())

        self.reranker_config = reranker_config or ReRankerConfig()
        self.prompt_builder = PromptBuilder(system_prompt)
        self.llm = OpenAIModel(client=OpenAI(api_key=api_key), model_name=model_name, system_prompt=system_prompt)

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
