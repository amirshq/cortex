"""
SQLite-backed chat history.

Survives server restarts because it writes to a file, unlike Redis which
resets when the TTL expires or the server bounces.

Roles
-----
  ensure_session   — create a session row if it doesn't already exist
  save_message     — append one user/assistant turn
  get_messages     — read messages for the history API (paginated)
  get_latest_summary — pull the last N messages from the user's most
                       recent session and return them as a plain-text
                       block so a new session can be hydrated with
                       prior context
  list_sessions    — list all sessions for a user
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ChatHistoryManager:
    def __init__(self, db_path: str = "data/chatbot.db") -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ setup
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id         TEXT PRIMARY KEY,
                    user_id    TEXT NOT NULL,
                    title      TEXT NOT NULL DEFAULT 'New conversation',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id    TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    timestamp  TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages (session_id, id);

                CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON sessions (user_id, created_at DESC);
            """)

    # --------------------------------------------------------------- sessions
    def ensure_session(
        self,
        session_id: str,
        user_id: str,
        title: str = "New conversation",
    ) -> None:
        """Create the session row only if it doesn't already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (id, user_id, title, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, user_id, title, datetime.utcnow().isoformat()),
            )

    def update_session_title(self, session_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )

    def list_sessions(self, user_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at
                FROM sessions
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------------- messages
    def save_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
    ) -> None:
        """Append one turn to the persistent history."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, user_id, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, role, content, datetime.utcnow().isoformat()),
            )

    def get_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict]:
        """Return messages for a session, oldest-first, with pagination."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY id
                LIMIT ? OFFSET ?
                """,
                (session_id, limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_messages(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row[0]

    # --------------------------------------------------------- session hydration
    def get_latest_summary(
        self,
        user_id: str,
        n_messages: int = 10,
    ) -> Optional[str]:
        """
        Pull the last N messages from the user's most-recent session and
        format them as a plain-text block.

        Returned string is injected as `chat_summary` when a brand-new
        session starts, so the model has prior context without the full
        history being replayed in the window.

        Returns None when the user has no previous history.
        """
        with self._connect() as conn:
            # Find the most recent session
            row = conn.execute(
                """
                SELECT id FROM sessions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

            if not row:
                return None

            session_id = row["id"]

            # Fetch the last N messages from that session
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, n_messages),
            ).fetchall()

        if not rows:
            return None

        # Reverse so they read oldest → newest
        lines = [
            f"{r['role'].capitalize()}: {r['content']}"
            for r in reversed(rows)
        ]
        return "\n".join(lines)
