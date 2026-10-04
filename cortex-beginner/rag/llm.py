"""Step 5 — The LLM engine: write an answer from the chunks.

This is the "G" in RAG (Retrieval-Augmented Generation). The LLM gets the
question plus the best chunks, and is told to answer ONLY from them. That
instruction is what stops it from inventing facts ("hallucinating").
"""

from openai import OpenAI

SYSTEM_PROMPT = (
    "You answer questions using only the context provided. "
    "If the answer is not in the context, reply exactly: I don't know."
)


class LLM:
    def __init__(self, model):
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model

    def answer(self, question, chunks):
        # Number the chunks so the model (and you, when debugging) can tell them apart.
        context = "\n\n".join(f"[{i}] {chunk['text']}" for i, chunk in enumerate(chunks, start=1))
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,  # always pick the most likely words: factual, repeatable answers
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content
