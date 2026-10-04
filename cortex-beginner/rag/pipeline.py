"""The whole RAG system: connect the five steps.

    INDEX (once):  documents → chunks → embed → store in the vector DB
    ASK   (each question):
        question → retrieve top-k → re-rank to top-n → LLM writes the answer
"""

import config
from rag.embedder import Embedder
from rag.llm import LLM
from rag.loader import load_documents, split_into_chunks
from rag.reranker import Reranker
from rag.retriever import Retriever
from rag.vector_db import VectorDB


class RAG:
    def __init__(self, embedder=None, vector_db=None, reranker=None, llm=None):
        # Each part can be passed in (handy for tests); otherwise use config.py.
        self.embedder = embedder or Embedder(config.EMBEDDING_MODEL)
        self.vector_db = vector_db or VectorDB(config.DB_DIR, config.COLLECTION)
        self.retriever = Retriever(self.embedder, self.vector_db)
        self.reranker = reranker or Reranker(config.RERANKER_MODEL)
        self.llm = llm or LLM(config.LLM_MODEL)

    def index(self, folder):
        """Load, chunk, embed and store every document in the folder."""
        documents = load_documents(folder)
        ids, texts, sources = [], [], []
        for doc in documents:
            for i, chunk in enumerate(split_into_chunks(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
                ids.append(f"{doc['source']}-{i}")   # e.g. "kestrel-specs.txt-0"
                texts.append(chunk)
                sources.append(doc["source"])
        if texts:
            self.vector_db.add(ids, texts, self.embedder.embed(texts), sources)
        return len(documents), len(texts)

    def ask(self, question):
        """Answer a question. Returns (answer, the chunks the answer was based on)."""
        candidates = self.retriever.retrieve(question, config.TOP_K)          # fast, rough
        best = self.reranker.rerank(question, candidates, config.TOP_N)       # slow, precise
        return self.llm.answer(question, best), best
