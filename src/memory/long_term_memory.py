# No specific long‑term memory service is hard‑coded in long_term_memory.py.
# Replace it with Letta Long Term Memory or Langchain long term memory
from typing import List, Dict, TYPE_CHECKING
from datetime import datetime

# Type checking only - avoids circular imports
if TYPE_CHECKING:
    from src.memory.vectordb import VectorDB


class LongTermMemory:
    """
    Cognitive long-term memory layer.
    Owns memory semantics, not storage.
    
    Uses dependency injection - vectordb, embedder, and chunker are passed in,not imported. 
    This makes the code more flexible and testable.
    """

    def __init__(
        self, 
        vectordb: "VectorDB",  # Type hint (string to avoid circular import)
        embedder,  # Object with embed() method
        chunker,   # Object with split() method
    ):
        """
        Initialize long-term memory.
        
        Args:
            vectordb: Vector database instance (from src.memory.vectordb.VectorDB)
            embedder: Embedding model (must have embed(text: str) or embed_query(text: str) method)
            chunker: Text chunker (must have split(text: str) method)
        """
        self.vectordb = vectordb
        self.embedder = embedder
        self.chunker = chunker

    def remember(
        self,
        content: str,
        user_id: str,
        memory_type: str = "knowledge",
        importance: int = 1,
    ) -> None:

        """
        chunk the information, embed each chunk, store in vectordb with metadata
        Store information in long-term memory using a pluggable chunking strategy.
        """
        chunks = self.chunker.split(content)

        for chunk in chunks:
            embedding = self.embedder.embed(chunk)

            self.vectordb.add(
                ids=[self._build_id(user_id)],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "user_id": user_id,
                    "type": memory_type,
                    "importance": importance,
                    "created_at": datetime.utcnow().isoformat(),
                }],
            )

    def recall(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Recall relevant memories for a user query.
        """
        embedding = self.embedder.embed(query)

        return self.vectordb.search(
            embedding=embedding,
            top_k=top_k,
            filters={"user_id": user_id},
        )

    def forget_user(self, user_id: str) -> None:
        """
        Delete all memories for a user.
        """
        self.vectordb.delete(filters={"user_id": user_id})

    @staticmethod
    def _build_id(user_id: str) -> str:
        return f"{user_id}-{datetime.utcnow().timestamp()}"