"""Every setting of the RAG system, in one place.

Change a model or a number here — nothing else needs to change.
"""

# --- Models -----------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"   # turns text into vectors (OpenAI)
LLM_MODEL = "gpt-4o-mini"                    # writes the final answer (OpenAI)
RERANKER_MODEL = "BAAI/bge-reranker-base"    # scores how relevant a chunk is (runs locally)

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # characters shared by neighbouring chunks, so no sentence is lost at a cut

# --- Retrieval --------------------------------------------------------------
TOP_K = 10   # how many chunks the vector database returns (fast, rough)
TOP_N = 3    # how many the re-ranker keeps for the LLM (slow, precise)

# --- Storage ----------------------------------------------------------------
DATA_DIR = "data"            # put your .txt / .md / .pdf files here
DB_DIR = "chroma_db"         # where the vector database is saved on disk
COLLECTION = "documents"     # the "table" name inside the vector database
