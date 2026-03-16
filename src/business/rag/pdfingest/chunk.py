# The main chunking logic is implemented here.

from dataclasses import dataclass
from importlib import metadata
from typing import List, Dict, Any
import hashlib

@dataclass
class Chunk:
    chunk_id: str
    text: str 
    metadata: Dict[str, Any]

class Chunker:
    """
    Deterministic text chunker with overlap.

    Single source of truth for chunking.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 100,
        strategy_name: str = "char_window_800_overlap_100",
    ):
        assert overlap < chunk_size, "overlap must be smaller than chunk_size"

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy_name = strategy_name

    def split(self,text: str, source_metadata: Dict,
    ) -> List[Chunk]:
        """
        Split raw text into overlapping chunks.
        Args:
            text: full document text
            source_metadata: metadata from ingestion (pdf path, page map, etc.)
        Returns:
            List[Chunk]
        """
        if not text or not text.strip():
            return []

        chunks: List[Chunk] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = self._build_chunk_id(chunk_text=chunk_text,start=start,source_metadata=source_metadata,)
                metadata = {**source_metadata,"chunk_start": start,"chunk_end": end,"chunk_strategy": self.strategy_name,}
                chunks.append(Chunk(chunk_id=chunk_id,text=chunk_text,metadata=metadata,))
            start = end - self.overlap
            if end >= text_length:
                break
        return chunks
    def _build_chunk_id(self,chunk_text: str,start: int,source_metadata: Dict,) -> str:
        """
        Build stable chunk ID 
        """
        base = (
            source_metadata.get("source_id", "")  #Pdf files Name
            + str(start)
            + chunk_text[:100]
        )
        return hashlib.sha1(base.encode("utf-8")).hexdigest()