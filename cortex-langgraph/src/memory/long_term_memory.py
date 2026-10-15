"""Long-term (semantic) memory — LangChain edition.

Same memory semantics as cortex-core (remember / recall / forget, scoped per
user), but the storage is any LangChain `VectorStore` instead of a hand-written
adapter, and chunking uses LangChain's `RecursiveCharacterTextSplitter`.

The vector store embeds text itself, so this class no longer needs an
embedder of its own — that dependency moved into the store.
"""

from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter


class LongTermMemory:
    """
    Cognitive long-term memory layer.
    Owns memory semantics, not storage.
    Uses dependency injection - the vector store (and optionally the text
    splitter) are passed in, not imported. This makes the code more flexible
    and testable.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        text_splitter: Optional[TextSplitter] = None,
    ):
        """
        Args:
            vectorstore: Any LangChain VectorStore that supports metadata
                         filtering on `user_id` (Chroma does).
            text_splitter: Used only by remember(); conversation turns are
                           stored whole. Defaults to a 1000-char recursive splitter.
        """
        self.vectorstore = vectorstore
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

    def remember(
        self,
        content: str,
        user_id: str,
        memory_type: str = "knowledge",
        importance: int = 1,
    ) -> None:
        """
        Store information in long-term memory: split it into chunks and store
        each chunk with user-scoped metadata (the store embeds them).
        """
        chunks = self.text_splitter.split_text(content)
        if not chunks:
            return
        docs = [
            Document(page_content=chunk, metadata=self._metadata(user_id, memory_type, importance))
            for chunk in chunks
        ]
        self.vectorstore.add_documents(docs, ids=[self._build_id(user_id, i) for i in range(len(docs))])

    def recall(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Recall relevant memories for a user query.

        Returns the same shape cortex-core did — [{"text", "metadata", "score"}] —
        so the prompt builder and the agent tool don't care which store is behind it.
        Score is the store's distance (lower = more similar).
        """
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k,
            filter={"user_id": user_id},
        )
        return [
            {"text": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results
        ]

    def forget_user(self, user_id: str) -> None:
        """
        Delete all memories for a user.

        LangChain's generic VectorStore.delete() only takes ids, so this looks
        the user's ids up first via Chroma's metadata filter (`get(where=...)`).
        """
        existing = self.vectorstore.get(where={"user_id": user_id})
        ids = existing.get("ids", [])
        if ids:
            self.vectorstore.delete(ids=ids)

    def remember_conversation(
        self,
        user_message: str,
        assistant_response: str,
        user_id: str,
    ) -> None:
        """
        Store a single conversation turn (user + assistant exchange) as one atomic unit.
        Skips chunking intentionally — conversation pairs are already small,
        and splitting them would break their semantic coherence.
        """
        text = f"user: {user_message}\nassistant: {assistant_response}"
        doc = Document(page_content=text, metadata=self._metadata(user_id, "conversation", 1))
        self.vectorstore.add_documents([doc], ids=[self._build_id(user_id)])

    @staticmethod
    def _metadata(user_id: str, memory_type: str, importance: int) -> Dict:
        return {
            "user_id": user_id,
            "type": memory_type,
            "importance": importance,
            "created_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def _build_id(user_id: str, index: int = 0) -> str:
        # The chunk index keeps ids unique when remember() stores several
        # chunks within the same timestamp tick.
        return f"{user_id}-{datetime.utcnow().timestamp()}-{index}"
