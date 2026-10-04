"""Ask a question: python ask.py "How long is the Kestrel-7 warranty?"

Without an argument it starts an interactive loop (empty line to quit).
"""

import sys

from dotenv import load_dotenv

load_dotenv()  # puts OPENAI_API_KEY from .env into the environment

from rag.pipeline import RAG

rag = RAG()


def show(question):
    answer, sources = rag.ask(question)
    print(f"\nAnswer: {answer}\n\nBased on:")
    for chunk in sources:
        print(f"  [{chunk['score']:.2f}] {chunk['source']}: {chunk['text'][:80]}...")


if len(sys.argv) > 1:
    show(" ".join(sys.argv[1:]))
else:
    while question := input("\nQuestion (empty to quit): ").strip():
        show(question)
