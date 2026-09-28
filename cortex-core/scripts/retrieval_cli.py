import sys
from pathlib import Path

# Ensure project root is on sys.path when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.business.rag.retrieval import RAGPipeline
from pathlib import Path
persist_dir = str(Path(__file__).resolve().parents[1] / "src/business/rag/vectorstore")
rag = RAGPipeline(persist_dir=persist_dir)
query = rag._retrieve("What are the unique total property counts for EPCs per year?", top_k=5)
if not query:
    print("No results retrieved")
else:
    print(query[0])