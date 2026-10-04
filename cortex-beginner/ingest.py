"""Build the index: python ingest.py

Reads every document in data/, chunks it, embeds it, and stores it in the
vector database. Run it again after adding or changing documents.
"""

from dotenv import load_dotenv

load_dotenv()  # puts OPENAI_API_KEY from .env into the environment

import config
from rag.pipeline import RAG

docs, chunks = RAG().index(config.DATA_DIR)
print(f"Indexed {docs} documents as {chunks} chunks into '{config.DB_DIR}/'.")
