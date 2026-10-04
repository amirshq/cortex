"""Step 3 — Retrieval: question → the most similar chunks.

Retrieval is just steps 1 and 2 together:
    1. embed the question (same embedding model as the documents!)
    2. ask the vector database for the nearest chunks

It is FAST but ROUGH: "similar meaning" is not always "answers the question".
That is why a re-ranker comes next.
"""


class Retriever:
    def __init__(self, embedder, vector_db):
        self.embedder = embedder
        self.vector_db = vector_db

    def retrieve(self, question, k):
        query_vector = self.embedder.embed([question])[0]
        return self.vector_db.search(query_vector, k)
