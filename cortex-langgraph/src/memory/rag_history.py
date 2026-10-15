"""
SQLite-backed history of RAG (PDF Q&A) questions and answers.

The chatbot's history lives in ChatHistoryManager's sessions/messages tables.
PDF Q&A is a different kind of record — one question, one answer, and the
source chunks that answer was based on — so it gets its own table in the same
database file instead of being squeezed into chat sessions (where it would
also show up in the chatbot's sidebar).

Roles
-----
  save   — append one question/answer (+ sources) for a user
  list   — a user's Q&A, oldest first, so the RAG panel can restore it
  clear  — delete a user's Q&A
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class RagHistoryManager:
    def __init__(self, db_path: str = "data/chatbot.db") -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS rag_history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL,
                    question   TEXT NOT NULL,
                    answer     TEXT NOT NULL,
                    sources    TEXT NOT NULL,      -- JSON list of {text, metadata, score}
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rag_history_user
                    ON rag_history (user_id, id);
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, user_id: str, question: str, answer: str, sources: List[Dict]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO rag_history (user_id, question, answer, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, question, answer, json.dumps(sources), datetime.utcnow().isoformat()),
            )

    def list(self, user_id: str, limit: int = 200) -> List[Dict]:
        """The user's most recent `limit` Q&A, returned oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, question, answer, sources, created_at FROM rag_history
                WHERE user_id = ? ORDER BY id DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "sources": json.loads(r["sources"]),
                "created_at": r["created_at"],
            }
            for r in reversed(rows)
        ]

    def clear(self, user_id: str) -> int:
        """Delete all of a user's Q&A. Returns how many rows were removed."""
        with self._connect() as conn:
            return conn.execute("DELETE FROM rag_history WHERE user_id = ?", (user_id,)).rowcount
