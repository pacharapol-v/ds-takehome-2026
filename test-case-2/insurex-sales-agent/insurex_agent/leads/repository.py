"""SQLite persistence for leads (used by the MCP server, or directly in tests)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .models import LeadRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    lead_id          TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    full_name        TEXT NOT NULL,
    occupation       TEXT NOT NULL,
    monthly_income   REAL NOT NULL CHECK (monthly_income > 0),
    income_text      TEXT,
    phone            TEXT NOT NULL CHECK (length(phone) BETWEEN 9 AND 10),
    product_interest TEXT,
    consent          INTEGER NOT NULL CHECK (consent = 1),
    status           TEXT NOT NULL DEFAULT 'new',
    source           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE (session_id, phone)
);
CREATE INDEX IF NOT EXISTS ix_leads_phone ON leads(phone);
CREATE TABLE IF NOT EXISTS lead_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id    TEXT NOT NULL REFERENCES leads(lead_id),
    event      TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LeadRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert(self, lead: LeadRecord) -> dict:
        """Idempotent: the same customer (session + phone) is updated, not duplicated,
        so a retried tool call can never create two leads."""
        now = _now()
        with self._conn() as c:
            row = c.execute("SELECT lead_id FROM leads WHERE session_id=? AND phone=?",
                            (lead.session_id, lead.phone)).fetchone()
            if row:
                lead_id, status = row["lead_id"], "updated"
                c.execute(
                    """UPDATE leads SET full_name=?, occupation=?, monthly_income=?, income_text=?,
                       product_interest=?, consent=?, updated_at=? WHERE lead_id=?""",
                    (lead.full_name, lead.occupation, lead.monthly_income, lead.income_text,
                     lead.product_interest, int(lead.consent), now, lead_id),
                )
            else:
                lead_id, status = f"LD-{uuid.uuid4().hex[:10].upper()}", "created"
                c.execute(
                    """INSERT INTO leads (lead_id, session_id, full_name, occupation, monthly_income, income_text,
                       phone, product_interest, consent, status, source, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (lead_id, lead.session_id, lead.full_name, lead.occupation, lead.monthly_income,
                     lead.income_text, lead.phone, lead.product_interest, int(lead.consent), "new",
                     lead.source, now, now),
                )
            c.execute("INSERT INTO lead_events (lead_id, event, payload, created_at) VALUES (?,?,?,?)",
                      (lead_id, status, lead.model_dump_json(), now))
        return {"lead_id": lead_id, "status": status}

    def get(self, lead_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM leads WHERE lead_id=?", (lead_id,)).fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 20, session_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM leads", []
        if session_id:
            q += " WHERE session_id=?"
            args.append(session_id)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def delete_by_session(self, session_id: str) -> list[str]:
        """Erase every lead of a session (PDPA right to erasure). The audit events
        hold a copy of the personal data, so they are deleted too. One transaction."""
        with self._conn() as c:
            ids = [r["lead_id"] for r in c.execute("SELECT lead_id FROM leads WHERE session_id=?", (session_id,))]
            c.executemany("DELETE FROM lead_events WHERE lead_id=?", [(i,) for i in ids])
            c.execute("DELETE FROM leads WHERE session_id=?", (session_id,))
        return ids

    def events(self, lead_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM lead_events WHERE lead_id=? ORDER BY id", (lead_id,)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]
