"""Application facade: wires brain + retriever + MCP gateway + checkpointer + sessions."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage

from ..config import ROOT, Settings, get_settings
from ..leads.gateway import LeadsGateway, make_gateway
from ..memory.sessions import SessionRegistry
from ..retrieval.hybrid import HybridRetriever
from .brain import DEGRADED_STEPS, make_brain
from .graph import SalesAgentGraph, welcome_message

log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    session_id: str
    user_message: str
    answer: str
    intent: str
    trace: list[str]
    mode: str
    lead_status: str
    lead: dict
    lead_id: str | None
    sources: list[str]
    active_products: list[str]
    retrieved: list[str] = field(default_factory=list)
    degraded_steps: list[str] = field(default_factory=list)
    latency_ms: int = 0


async def forget_session(session_id: str, checkpointer, gateway: LeadsGateway, registry: SessionRegistry) -> dict:
    """Erase everything stored for one session: its leads (through the MCP tool), the
    checkpointed conversation and the registry row. Leads go first, so if the tool call
    fails nothing else is deleted and the request can simply be retried."""
    out = await gateway.delete_leads(session_id)
    if not out.get("ok"):
        raise RuntimeError(f"could not delete leads of {session_id}: {out}")
    await checkpointer.adelete_thread(session_id)
    known = registry.delete(session_id)
    log.info("forgot session %s (leads deleted: %s)", session_id, out["deleted"])
    return {"session_id": session_id, "leads_deleted": out["deleted"], "session_existed": known}


class SalesAgent:
    def __init__(self, settings: Settings, brain, retriever: HybridRetriever, gateway: LeadsGateway,
                 checkpointer, registry: SessionRegistry, turn_log: Path | None = None, saver_cm=None):
        self.s = settings
        self.brain = brain
        self.retriever = retriever
        self.gateway = gateway
        self.checkpointer = checkpointer
        self.registry = registry
        self.turn_log = turn_log
        self._saver_cm = saver_cm
        self.graph = SalesAgentGraph(brain, retriever, gateway, settings).build(checkpointer)

    @classmethod
    async def create(cls, settings: Settings | None = None, *, brain=None, retriever=None, gateway=None,
                     checkpointer=None, turn_log: str | Path | None = "logs/turns.jsonl") -> "SalesAgent":
        s = settings or get_settings()
        if brain is None:
            from ..llm import check_credentials
            check_credentials(s, need_llm=True)
        saver_cm = None
        if checkpointer is None:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            saver_cm = AsyncSqliteSaver.from_conn_string(str(s.checkpoint_db_path))
            checkpointer = await saver_cm.__aenter__()
        tl = Path(turn_log) if turn_log else None
        if tl and not tl.is_absolute():
            tl = ROOT / tl
        if tl:
            tl.parent.mkdir(parents=True, exist_ok=True)
        retriever = retriever or HybridRetriever(settings=s)
        brain = brain or make_brain(s)
        rules = getattr(brain, "fallback", brain)          # rule brain (direct or as LLM fallback)
        if hasattr(rules, "bind_corpus"):
            rules.bind_corpus([c.text for c in retriever.kb.children])
        return cls(s, brain, retriever, gateway or make_gateway(s), checkpointer,
                   SessionRegistry(s.sessions_db_path), tl, saver_cm)

    def _config(self, session_id: str, user_id: str | None) -> dict:
        return {"configurable": {"thread_id": session_id}, "metadata": {"user_id": user_id or "anonymous"},
                "recursion_limit": 40}

    async def chat(self, session_id: str, message: str, user_id: str | None = None) -> TurnResult:
        t0 = time.perf_counter()
        degraded: list[str] = []
        token = DEGRADED_STEPS.set(degraded)
        try:
            self.registry.touch(session_id, user_id)
            out = await self.graph.ainvoke(
                {"messages": [HumanMessage(message)], "session_id": session_id},
                self._config(session_id, user_id),
            )
        finally:
            DEGRADED_STEPS.reset(token)
        res = TurnResult(
            session_id=session_id, user_message=message, answer=out.get("answer", ""),
            intent=(out.get("route") or {}).get("intent", "-"), trace=out.get("turn_trace", []),
            mode=out.get("mode", "qa"), lead_status=out.get("lead_status", "none"), lead=out.get("lead") or {},
            lead_id=out.get("lead_id"), sources=out.get("sources", []),
            active_products=out.get("active_products", []),
            retrieved=[r["citation"] for r in (out.get("relevant") or out.get("retrieved") or [])],
            degraded_steps=degraded, latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        if self.turn_log:
            with open(self.turn_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
        return res

    async def get_state(self, session_id: str) -> dict:
        snap = await self.graph.aget_state(self._config(session_id, None))
        return snap.values if snap else {}

    async def history(self, session_id: str) -> list[tuple[str, str]]:
        vals = await self.get_state(session_id)
        return [("user" if isinstance(m, HumanMessage) else "assistant", str(m.content))
                for m in vals.get("messages", [])]

    async def welcome(self, session_id: str) -> str | None:
        """The introduction for a session that has no conversation yet; None when resuming one."""
        return None if (await self.get_state(session_id)).get("messages") else welcome_message()

    async def forget(self, session_id: str) -> dict:
        return await forget_session(session_id, self.checkpointer, self.gateway, self.registry)

    async def aclose(self) -> None:
        await self.gateway.aclose()
        if self._saver_cm is not None:
            await self._saver_cm.__aexit__(None, None, None)
