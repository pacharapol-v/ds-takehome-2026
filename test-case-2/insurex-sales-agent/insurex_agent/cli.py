"""Command-line interface.

    python -m insurex_agent.cli chat [--session ID] [--user USER] [--debug]
    python -m insurex_agent.cli ask "คำถาม" [--session ID]
    python -m insurex_agent.cli leads [--limit N]
    python -m insurex_agent.cli sessions
    python -m insurex_agent.cli forget ID [--yes]   # erase a session + its leads (PDPA)
    python -m insurex_agent.cli graph            # print the LangGraph as Mermaid

In-chat commands: /new  /switch <id>  /history  /state  /leads  /sessions  /forget [id]  /help  /quit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .memory.sessions import SessionRegistry, new_session_id

console = Console()


def _print_turn(res, debug: bool) -> None:
    console.print(Panel(res.answer, title="InsureX Assistant", border_style="magenta", expand=False))
    if debug:
        console.print(f"[dim]intent={res.intent} | trace={' → '.join(res.trace)} | mode={res.mode} | "
                      f"lead={res.lead_status} {res.lead_id or ''} | {res.latency_ms} ms"
                      + (f" | DEGRADED: {res.degraded_steps}" if res.degraded_steps else "") + "[/dim]")


async def _leads_table(agent_or_gateway, limit: int = 10) -> None:
    gw = getattr(agent_or_gateway, "gateway", agent_or_gateway)
    out = await gw.list_leads(limit=limit)
    t = Table(title="Leads (via MCP list_leads)")
    for col in ("lead_id", "session_id", "full_name", "occupation", "monthly_income", "phone", "product_interest",
                "created_at"):
        t.add_column(col)
    for r in out.get("leads", []):
        t.add_row(*(str(r.get(c, "")) for c in ("lead_id", "session_id", "full_name", "occupation", "monthly_income",
                                                 "phone", "product_interest", "created_at")))
    console.print(t)


async def _print_welcome(agent, sid: str) -> None:
    text = await agent.welcome(sid)
    if text:
        console.print(Panel(text, title="InsureX Assistant", border_style="magenta", expand=False))


def _confirm_forget(sid: str) -> bool:
    console.print(f"[bold red]Erase session {sid}[/bold red]: its conversation and all its leads. This cannot be undone.")
    try:
        # lstrip: Windows PowerShell prefixes piped input with a byte-order mark
        return console.input("Type 'yes' to confirm: ").lstrip("﻿").strip().lower() == "yes"
    except (EOFError, KeyboardInterrupt):
        return False


def _print_forgot(out: dict) -> None:
    leads = ", ".join(out["leads_deleted"]) or "none"
    console.print(f"forgot session [cyan]{out['session_id']}[/cyan] (registered: {out['session_existed']}); "
                  f"leads deleted: {leads}")


async def chat(session: str | None, user: str | None, debug: bool) -> None:
    from .agent.app import SalesAgent

    s = get_settings()
    console.print(f"[bold]InsureX Sales Agent[/bold]  llm={s.llm_provider}:{s.llm_model}  "
                  f"embeddings={s.embedding_provider}:{s.embedding_model}  leads={s.leads_backend}")
    agent = await SalesAgent.create(s)
    sid = session or new_session_id()
    console.print(f"session: [cyan]{sid}[/cyan]   (type /help for commands)")
    await _print_welcome(agent, sid)
    try:
        while True:
            try:
                text = console.input("[bold green]คุณ ›[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text.startswith("/"):
                cmd, *arg = text.split(maxsplit=1)
                if cmd in ("/quit", "/exit"):
                    break
                if cmd == "/new":
                    sid = new_session_id()
                    console.print(f"new session: [cyan]{sid}[/cyan]")
                    await _print_welcome(agent, sid)
                elif cmd == "/switch" and arg:
                    sid = arg[0]
                    console.print(f"switched to session: [cyan]{sid}[/cyan]")
                    await _print_welcome(agent, sid)
                elif cmd == "/history":
                    for role, content in await agent.history(sid):
                        console.print(f"[bold]{role}[/bold]: {content}")
                elif cmd == "/state":
                    st = await agent.get_state(sid)
                    keep = {k: st.get(k) for k in ("mode", "lead_status", "lead", "lead_id", "active_products",
                                                    "summary", "language")}
                    keep["n_messages"] = len(st.get("messages", []))
                    console.print_json(json.dumps(keep, ensure_ascii=False, default=str))
                elif cmd == "/leads":
                    await _leads_table(agent)
                elif cmd == "/sessions":
                    for r in agent.registry.list():
                        console.print(r)
                elif cmd == "/forget":
                    target = arg[0] if arg else sid
                    if _confirm_forget(target):
                        _print_forgot(await agent.forget(target))
                        if target == sid:
                            sid = new_session_id()
                            console.print(f"new session: [cyan]{sid}[/cyan]")
                            await _print_welcome(agent, sid)
                    else:
                        console.print("cancelled - nothing deleted")
                else:
                    console.print(__doc__)
                continue
            res = await agent.chat(sid, text, user_id=user)
            _print_turn(res, debug)
    finally:
        await agent.aclose()


async def ask(question: str, session: str | None, debug: bool) -> None:
    from .agent.app import SalesAgent

    agent = await SalesAgent.create()
    try:
        res = await agent.chat(session or new_session_id("ask"), question)
        _print_turn(res, debug)
    finally:
        await agent.aclose()


async def leads(limit: int) -> None:
    from .leads.gateway import make_gateway

    gw = make_gateway()
    try:
        await _leads_table(gw, limit)
    finally:
        await gw.aclose()


async def forget(session: str, yes: bool) -> None:
    """Erase a session without building the agent (no API key or index needed)."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from .agent.app import forget_session
    from .leads.gateway import make_gateway

    if not (yes or _confirm_forget(session)):
        console.print("cancelled - nothing deleted")
        return
    s = get_settings()
    gw = make_gateway(s)
    try:
        async with AsyncSqliteSaver.from_conn_string(str(s.checkpoint_db_path)) as saver:
            _print_forgot(await forget_session(session, saver, gw, SessionRegistry(s.sessions_db_path)))
    finally:
        await gw.aclose()


def graph_mermaid() -> str:
    from .agent.graph import SalesAgentGraph

    class _Dummy:  # graph structure does not depend on the dependencies
        pass

    return SalesAgentGraph(_Dummy(), _Dummy(), _Dummy()).build().get_graph().draw_mermaid()


def main() -> None:
    ap = argparse.ArgumentParser(prog="insurex_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("chat")
    p.add_argument("--session")
    p.add_argument("--user")
    p.add_argument("--debug", action="store_true")
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--session")
    p.add_argument("--debug", action="store_true")
    p = sub.add_parser("leads")
    p.add_argument("--limit", type=int, default=20)
    sub.add_parser("sessions")
    p = sub.add_parser("forget")
    p.add_argument("session")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    sub.add_parser("graph")
    args = ap.parse_args()
    level = get_settings().log_level
    if args.cmd in ("chat", "ask") and not args.debug:
        level = "WARNING"           # clean demo: only the conversation (+ warnings/errors); --debug shows the logs
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("httpx", "httpx2", "mcp", "chromadb", "langchain_mcp_adapters"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "chat":
        asyncio.run(chat(args.session, args.user, args.debug))
    elif args.cmd == "ask":
        asyncio.run(ask(args.question, args.session, args.debug))
    elif args.cmd == "leads":
        asyncio.run(leads(args.limit))
    elif args.cmd == "sessions":
        for r in SessionRegistry(get_settings().sessions_db_path).list(100):
            print(json.dumps(r, ensure_ascii=False))
    elif args.cmd == "forget":
        asyncio.run(forget(args.session, args.yes))
    elif args.cmd == "graph":
        print(graph_mermaid())


if __name__ == "__main__":
    main()
