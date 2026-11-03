"""Session management.

Two layers:
1. **Conversation state** - LangGraph checkpointer (SQLite). Every session is a
   `thread_id`; the full `AgentState` (messages, running summary, lead draft,
   products in focus ...) is checkpointed after every node, so a session survives
   process restarts and two sessions can never see each other's context.
2. **Session registry** - a small table with who/when metadata so sessions can
   be listed, resumed and expired (the checkpointer itself is not queryable
   that way).

Efficient recall: the prompt only receives the last `HISTORY_WINDOW` messages
plus a running summary; once a thread exceeds `SUMMARIZE_AFTER` messages the
`prepare_turn` node folds the oldest ones into the summary and deletes them
from state (RemoveMessage), so token cost per turn stays flat.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    user_id        TEXT,
    channel        TEXT NOT NULL DEFAULT 'cli',
    created_at     TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    turns          INTEGER NOT NULL DEFAULT 0
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id(prefix: str = "sess") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class SessionRegistry:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def touch(self, session_id: str, user_id: str | None = None, channel: str = "cli") -> None:
        now = _now()
        with self._conn() as c:
            c.execute(
                """INSERT INTO sessions (session_id, user_id, channel, created_at, last_active_at, turns)
                   VALUES (?,?,?,?,?,1)
                   ON CONFLICT(session_id) DO UPDATE SET last_active_at=excluded.last_active_at,
                       turns=sessions.turns+1, user_id=COALESCE(excluded.user_id, sessions.user_id)""",
                (session_id, user_id, channel, now, now),
            )

    def get(self, session_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def delete(self, session_id: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,)).rowcount > 0

    def list(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM sessions ORDER BY last_active_at DESC LIMIT ?", (limit,)).fetchall()]
