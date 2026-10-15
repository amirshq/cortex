import sys
from pathlib import Path

# Ensure project root is on sys.path when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.business.rag.vector_store import create_vector_store
from pathlib import Path
persist_dir = str(Path(__file__).resolve().parents[1] / "src/business/rag/vectorstore")
store = create_vector_store(persist_dir=persist_dir)
# Retrieval only (no re-rank, no LLM): the vector store's raw top-5.
results = store.similarity_search_with_score("What are the unique total property counts for EPCs per year?", k=5)
if not results:
    print("No results retrieved")
else:
    doc, distance = results[0]
    print(f"distance={distance:.4f} metadata={doc.metadata}\n{doc.page_content}")