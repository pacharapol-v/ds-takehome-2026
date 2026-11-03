"""Client side of the lead tools.

`MCPLeadsGateway`  - spawns the MCP server (stdio) and calls its tools through
                     langchain-mcp-adapters. One long-lived MCP session is
                     reused for the whole app run.
`DirectLeadsGateway` - same interface, in-process repository (unit tests,
                     or when MCP is disabled via LEADS_BACKEND=direct).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Protocol

from ..config import ROOT, Settings, get_settings
from .models import LeadRecord
from .repository import LeadRepository

log = logging.getLogger(__name__)


class LeadsGateway(Protocol):
    async def save_lead(self, record: LeadRecord) -> dict: ...
    async def list_leads(self, limit: int = 20, session_id: str | None = None) -> dict: ...
    async def delete_leads(self, session_id: str) -> dict: ...
    async def aclose(self) -> None: ...


def _parse_tool_output(out: Any) -> dict:
    """langchain-mcp-adapters returns str, a list of content blocks, or (content, artifact)."""
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        return out
    if isinstance(out, list):
        out = "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in out)
    try:
        return json.loads(out)
    except Exception:
        return {"ok": False, "errors": [f"unparseable tool output: {str(out)[:200]}"]}


class MCPLeadsGateway:
    """The MCP stdio session is owned by one background task: anyio requires the
    session context to be entered and exited in the same task, while LangGraph
    runs nodes in their own tasks. Tool calls themselves may come from any task."""

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._tools: dict = {}
        self._ready: asyncio.Event | None = None
        self._closing: asyncio.Event | None = None
        self._runner: asyncio.Task | None = None
        self._error: BaseException | None = None
        self._lock = asyncio.Lock()

    def _connection(self) -> dict:
        env = {**os.environ, "LEADS_DB_PATH": str(self.s.leads_db_path), "PYTHONPATH": str(ROOT)}
        return {"command": sys.executable, "args": ["-m", "insurex_agent.leads.mcp_server"],
                "transport": "stdio", "env": env, "cwd": str(ROOT)}

    async def _run(self) -> None:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        try:
            client = MultiServerMCPClient({"leads": self._connection()})
            async with client.session("leads") as session:
                tools = await load_mcp_tools(session)
                self._tools = {t.name: t for t in tools}
                log.info("MCP leads server connected; tools=%s", sorted(self._tools))
                self._ready.set()
                await self._closing.wait()
        except BaseException as e:  # surface start-up failures to the caller
            self._error = e
            self._ready.set()
        finally:
            self._tools = {}

    async def start(self) -> None:
        async with self._lock:
            if self._tools:
                return
            if self._runner and self._runner.done():
                self._runner = None
            if self._runner is None:
                self._error = None
                self._ready, self._closing = asyncio.Event(), asyncio.Event()
                self._runner = asyncio.create_task(self._run(), name="mcp-leads-session")
            await asyncio.wait_for(self._ready.wait(), timeout=30)
            if self._error:
                err, self._runner = self._error, None
                raise RuntimeError(f"could not start MCP leads server: {err}") from err

    async def _call(self, name: str, args: dict) -> dict:
        await self.start()
        out = await self._tools[name].ainvoke(args)
        return _parse_tool_output(out)

    async def save_lead(self, record: LeadRecord) -> dict:
        args = record.model_dump(include={"session_id", "full_name", "occupation", "monthly_income", "phone",
                                          "consent", "product_interest", "income_text"})
        return await self._call("save_lead", args)

    async def list_leads(self, limit: int = 20, session_id: str | None = None) -> dict:
        args: dict = {"limit": limit}
        if session_id:
            args["session_id"] = session_id
        return await self._call("list_leads", args)

    async def delete_leads(self, session_id: str) -> dict:
        return await self._call("delete_leads", {"session_id": session_id})

    async def tool_names(self) -> list[str]:
        await self.start()
        return sorted(self._tools)

    async def aclose(self) -> None:
        if self._runner and not self._runner.done():
            self._closing.set()
            try:
                await asyncio.wait_for(self._runner, timeout=10)
            except Exception:  # pragma: no cover
                self._runner.cancel()
        self._runner = None


class DirectLeadsGateway:
    def __init__(self, settings: Settings | None = None, repo: LeadRepository | None = None):
        self.s = settings or get_settings()
        self.repo = repo or LeadRepository(self.s.leads_db_path)

    async def save_lead(self, record: LeadRecord) -> dict:
        return {"ok": True, **self.repo.upsert(record)}

    async def list_leads(self, limit: int = 20, session_id: str | None = None) -> dict:
        return {"ok": True, "leads": self.repo.list(limit, session_id)}

    async def delete_leads(self, session_id: str) -> dict:
        return {"ok": True, "deleted": self.repo.delete_by_session(session_id)}

    async def aclose(self) -> None:
        return None


def make_gateway(settings: Settings | None = None) -> LeadsGateway:
    s = settings or get_settings()
    return MCPLeadsGateway(s) if s.leads_backend == "mcp" else DirectLeadsGateway(s)
