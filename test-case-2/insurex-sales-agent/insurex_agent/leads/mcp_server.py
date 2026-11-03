"""MCP server exposing lead storage as tools (Model Context Protocol, stdio transport).

Run standalone:   python -m insurex_agent.leads.mcp_server
Inspect:          npx @modelcontextprotocol/inspector python -m insurex_agent.leads.mcp_server

The agent talks to this server through `langchain-mcp-adapters`; any other MCP
client (Claude Desktop, an internal sales console, ...) can use the same tools,
e.g. `list_leads` for the sales team's follow-up queue.

The server re-validates every payload with the same Pydantic contract the agent
uses, so a buggy or compromised client still cannot write malformed data.
"""

from __future__ import annotations

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from .models import LeadRecord
from .repository import LeadRepository

mcp = FastMCP("insurex-leads", log_level="WARNING")
_repo: LeadRepository | None = None


def repo() -> LeadRepository:
    global _repo
    if _repo is None:
        from ..config import get_settings
        path = os.environ.get("LEADS_DB_PATH") or str(get_settings().leads_db_path)
        _repo = LeadRepository(path)
    return _repo


@mcp.tool()
def save_lead(
    session_id: str,
    full_name: str,
    occupation: str,
    monthly_income: float,
    phone: str,
    consent: bool,
    product_interest: Optional[str] = None,
    income_text: Optional[str] = None,
) -> dict:
    """Save (or update) a sales lead captured in a chat session.

    All four customer fields are mandatory. `consent` must be true - the customer
    explicitly agreed to be contacted (PDPA). Saving the same session+phone twice
    updates the existing lead (idempotent). Returns {ok, lead_id, status} or
    {ok: false, errors}.
    """
    try:
        record = LeadRecord(
            session_id=session_id, full_name=full_name, occupation=occupation, monthly_income=monthly_income,
            phone=phone, consent=consent, product_interest=product_interest, income_text=income_text,
        )
    except ValidationError as e:
        return {"ok": False, "errors": [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]}
    return {"ok": True, **repo().upsert(record)}


@mcp.tool()
def get_lead(lead_id: str) -> dict:
    """Fetch one lead by id."""
    row = repo().get(lead_id)
    return {"ok": bool(row), "lead": row}


@mcp.tool()
def list_leads(limit: int = 20, session_id: Optional[str] = None) -> dict:
    """List the most recent leads (optionally for one chat session)."""
    return {"ok": True, "leads": repo().list(limit=limit, session_id=session_id)}


@mcp.tool()
def delete_leads(session_id: str) -> dict:
    """Erase all leads (and their audit events) captured in one chat session - the
    customer's PDPA right to erasure. Returns {ok, deleted: [lead_id, ...]}."""
    return {"ok": True, "deleted": repo().delete_by_session(session_id)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
