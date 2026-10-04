"""Step 2 — The vector database: store vectors, find the closest ones.

A vector database stores each chunk next to its vector. Given a new vector
(the question's), it quickly finds the stored vectors nearest to it — i.e.
the chunks whose meaning is closest to the question.

We use Chroma: it runs inside Python and saves to a folder, no server needed.
"""

import chromadb


class VectorDB:
    def __init__(self, path, collection_name):
        client = chromadb.PersistentClient(path=path)  # saved to disk, survives restarts
        self.collection = client.get_or_create_collection(collection_name)

    def add(self, ids, texts, vectors, sources):
        """Store chunks with their vectors. Re-using an id overwrites that chunk."""
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=[{"source": source} for source in sources],
        )

    def search(self, query_vector, k):
        """Return the k chunks closest to query_vector, closest first."""
        result = self.collection.query(query_embeddings=[query_vector], n_results=k)
        # Chroma returns one list per query; we sent one query, so take [0].
        return [
            {"text": text, "source": meta["source"], "distance": distance}
            for text, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def count(self):
        return self.collection.count()
