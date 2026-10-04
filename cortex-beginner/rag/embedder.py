"""Step 1 — The embedding model: text → vector.

An embedding is a list of numbers (here 1536 of them) that captures the
MEANING of a text. Texts with similar meaning get similar vectors, so
"How long is the warranty?" lands close to "Warranty covers 36 months" —
even though they share almost no words.
"""

from openai import OpenAI


class Embedder:
    def __init__(self, model):
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model

    def embed(self, texts):
        """Turn a list of texts into a list of vectors (one API call for all of them)."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
