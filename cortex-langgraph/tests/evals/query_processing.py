"""Query processing for experiment 3.5: change WHAT is searched, not what is answered.

The RAG graph's retrieve node calls vector_store.similarity_search_with_score(question).
QueryProcessingStore wraps the real store and transforms the query first, so the
real graph runs unchanged and the LLM still answers the user's ORIGINAL question.

Modes:
  rewrite  one LLM call: the question → a search query in the vocabulary an
           expert author would use (paraphrased questions miss on vocabulary).
  hyde     Hypothetical Document Embeddings (Gao et al., 2022): the LLM writes a
           short passage that WOULD answer the question; that passage is embedded
           and searched, since it looks more like a book chunk than a question does.
  multi    the LLM writes 3 rephrasings; the original + each rephrasing are searched
           and the result lists are merged with reciprocal rank fusion (RRF).

Prompts deliberately name no domain and give no examples: examples drawn from
the book would leak golden-set answers into retrieval.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import List

from langchain_core.documents import Document

MODES = ("none", "rewrite", "hyde", "multi")

REWRITE_PROMPT = """Rewrite the user's question as ONE search query for finding the passage
that answers it in a collection of technical documents. Use the precise technical
terms an expert author would use for the concepts the question describes, even if
the question describes them in everyday words. Keep every name, number and date.
Return only the query, nothing else."""

HYDE_PROMPT = """Write a short passage (3 to 4 sentences) that could appear in a technical
book and that answers the question. Write it the way an expert author would, using
the standard technical terminology. If you are unsure of exact facts, write a
plausible passage anyway: it is only used to search for the real one.
Return only the passage."""

MULTI_PROMPT = """Write 3 different search queries for finding the passage that answers the
user's question in a collection of technical documents. Each query should describe
the question's concepts with DIFFERENT wording, including the precise technical
terms an expert author would use. Keep every name, number and date.
Return exactly 3 lines, one query per line, with no numbering."""

RRF_K = 60  # the standard reciprocal-rank-fusion constant (Cormack et al., 2009)


class QueryProcessingStore:
    """Wraps a VectorStore; transforms the query before similarity search.

    Everything else (get, add_documents, ...) is passed through to the real store.
    `llm_seconds` is the time spent in the query LLM call for the last question,
    so the evaluator can report it as its own latency stage.
    """

    def __init__(self, store, llm, mode: str):
        if mode not in MODES:
            raise ValueError(f"Unknown query mode {mode!r}. Use one of {MODES}.")
        self.store, self.llm, self.mode = store, llm, mode
        self.llm_seconds = 0.0
        self.last_queries: List[str] = []

    def __getattr__(self, name):
        return getattr(self.store, name)

    def _ask(self, system: str, question: str) -> str:
        t0 = time.perf_counter()
        try:
            return self.llm.invoke([("system", system), ("human", question)]).content.strip()
        finally:
            self.llm_seconds += time.perf_counter() - t0

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        self.llm_seconds = 0.0
        if self.mode == "none":
            self.last_queries = [query]
            return self.store.similarity_search_with_score(query, k=k, **kwargs)
        if self.mode == "rewrite":
            self.last_queries = [self._ask(REWRITE_PROMPT, query)]
            return self.store.similarity_search_with_score(self.last_queries[0], k=k, **kwargs)
        if self.mode == "hyde":
            self.last_queries = [self._ask(HYDE_PROMPT, query)]
            return self.store.similarity_search_with_score(self.last_queries[0], k=k, **kwargs)
        # multi: original + 3 rephrasings, fused by rank
        variants = [line.strip() for line in self._ask(MULTI_PROMPT, query).splitlines() if line.strip()][:3]
        self.last_queries = [query] + variants
        return rrf_fuse([self.store.similarity_search_with_score(q, k=k, **kwargs) for q in self.last_queries], k)


def rrf_fuse(result_lists, k: int):
    """Reciprocal rank fusion: score(doc) = Σ 1 / (RRF_K + rank). Returns (doc, score)
    pairs, best first. The score is stored where a vector distance would be; it is
    only used for ordering (and only by the re-ranker's optional blending)."""
    scores, docs = defaultdict(float), {}
    for results in result_lists:
        for rank, (doc, _) in enumerate(results, start=1):
            key = doc.id or doc.page_content
            scores[key] += 1 / (RRF_K + rank)
            docs[key] = doc
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [(Document(id=docs[key].id, page_content=docs[key].page_content, metadata=docs[key].metadata),
             scores[key]) for key in best]
