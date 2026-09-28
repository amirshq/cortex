"""Quick diagnostic: check vectorstore contents and embedder work correctly."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
import os

load_dotenv()

# 1. Check API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"1. OPENAI_API_KEY loaded: {'YES' if api_key else 'NO'}")
if api_key:
    print(f"   Key starts with: {api_key[:12]}...")

# 2. Check vectorstore
persist_dir = str(ROOT / "src" / "business" / "rag" / "vectorstore")
print(f"\n2. Vector store path: {persist_dir}")
print(f"   Exists: {Path(persist_dir).exists()}")

import chromadb
client = chromadb.PersistentClient(path=persist_dir)
collections = client.list_collections()
print(f"   Collections: {[c.name for c in collections]}")

for col in collections:
    count = col.count()
    print(f"   Collection '{col.name}': {count} records")
    if count > 0:
        peek = col.peek(limit=2)
        print(f"   Sample IDs: {peek['ids'][:2]}")
        print(f"   Sample doc preview: {peek['documents'][0][:100] if peek['documents'] else 'EMPTY'}...")

# 3. Check embedder
print("\n3. Testing embedder...")
try:
    from src.business.core.embedding import OpenAIEmbedder
    embedder = OpenAIEmbedder(api_key=api_key)
    result = embedder.embed_query("test query")
    print(f"   Embedding dimension: {len(result)}")
    print(f"   First 5 values: {result[:5]}")
    print("   Embedder: OK")
except Exception as e:
    print(f"   Embedder ERROR: {e}")
