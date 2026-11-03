"""Scripted demo producing the run logs required by the brief.

    python scripts/run_demo.py

Scenarios (two customers interleaved, then a process restart):
  1. Alice (Thai)  - product Q&A, follow-up with a pronoun, comparison, interest -> lead
                     collection with a Q&A detour, an invalid phone number that is
                     rejected, correction, PDPA confirmation, save via MCP tool.
  2. Bob (English) - interleaved with Alice: his "this plan" must resolve to HIS product.
  3. Error handling - question outside the knowledge base, figure not in the brochure,
                     off-topic question.
  4. Restart       - a new agent process resumes Alice's session from SQLite.
  5. Leads table   - read back through the MCP `list_leads` tool.

Outputs
  logs/demo_<llm>.log    human-readable transcript with node traces
  logs/turns.jsonl       one JSON record per turn (structured log)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from insurex_agent.agent.app import SalesAgent  # noqa: E402
from insurex_agent.config import get_settings  # noqa: E402

ALICE = "alice-th-001"
BOB = "bob-en-002"
ERR = "edge-cases-003"

SCRIPT_1 = [
    (ALICE, "สวัสดีครับ"),
    (ALICE, "ตลอดชีพ 90/20 ผู้หญิงอายุ 35 ทุน 500,000 ต้องจ่ายเบี้ยปีละเท่าไหร่ครับ"),
    (BOB, "Hi, what is the annual premium of Whole Life 99/99 for a 40-year-old man?"),
    (ALICE, "แล้วแผนนี้รับประกันอายุถึงเท่าไหร่ จ่ายรายเดือนได้ไหม"),
    (BOB, "What is the maximum entry age for this plan?"),
    (ALICE, "เปรียบเทียบกับตลอดชีพ 99/99 หน่อย ต่างกันยังไง"),
    (ALICE, "สนใจ 90/20 ครับ อยากให้ตัวแทนติดต่อกลับ"),
    (BOB, "I'm interested. My name is John Smith, I'm a software engineer."),
    (ALICE, "ผมชื่อ สมชาย ใจดี อาชีพ วิศวกร แล้วแผนนี้ลดหย่อนภาษีได้ไหม"),
    (ALICE, "เงินเดือน 5 หมื่นครับ เบอร์ 08123"),
    (BOB, "My income is 80,000 baht per month and my phone is +66 89 765 4321"),
    (ALICE, "ขอโทษครับ เบอร์ที่ถูกคือ 081-234-5678"),
    (BOB, "yes"),
    (ALICE, "ยืนยันครับ"),
]
SCRIPT_ERR = [
    (ERR, "ประกันรถยนต์ชั้น 1 ราคาเท่าไหร่"),
    (ERR, "คุ้มชีวา 90/10 ผู้หญิงอายุ 25 เบี้ยเท่าไหร่"),
    (ERR, "ช่วยเขียนโค้ด Python ให้หน่อย"),
]
SCRIPT_RESTART = [
    (ALICE, "ขอบคุณครับ อีกเรื่อง ถ้าผมเสียชีวิตตอนอายุ 60 ครอบครัวได้เงินเท่าไหร่"),
]


class Transcript:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")

    def write(self, text: str = "") -> None:
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def turn(self, res) -> None:
        self.write(f"\n[{res.session_id}] USER: {res.user_message}")
        self.write(f"[{res.session_id}] BOT : " + res.answer.replace("\n", "\n" + " " * (len(res.session_id) + 9)))
        meta = {"intent": res.intent, "trace": " → ".join(res.trace), "mode": res.mode, "lead_status": res.lead_status,
                "lead": res.lead or None, "lead_id": res.lead_id, "active_products": res.active_products,
                "latency_ms": res.latency_ms}
        if res.degraded_steps:
            meta["DEGRADED (rule fallback)"] = res.degraded_steps
        self.write("          " + json.dumps(meta, ensure_ascii=False))


async def run_script(agent: SalesAgent, script, tr: Transcript) -> None:
    for sid, msg in script:
        tr.turn(await agent.chat(sid, msg))


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    s = get_settings()
    tag = "offline-rules" if s.llm_provider == "offline" else f"{s.llm_provider}-{s.llm_model}"
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    tr = Transcript(logs / f"demo_{tag.replace('/', '_').replace(':', '_')}.log")
    tr.write(f"# InsureX Sales Agent demo — {datetime.now().isoformat(timespec='seconds')}")
    tr.write(f"# LLM: {s.llm_provider}:{s.llm_model} | embeddings: {s.embedding_provider}:{s.embedding_model} | "
             f"leads backend: {s.leads_backend} | state dir: {s.state_dir}")
    if s.llm_provider == "offline":
        tr.write("# NOTE: offline mode = rule-based brain (no LLM). Answers are extractive quotes from the brochures.")

    tr.write("\n## 1+2. Two customers, interleaved sessions (Thai lead journey + English)")
    agent = await SalesAgent.create(s)
    await run_script(agent, SCRIPT_1, tr)

    tr.write("\n## 3. Error handling")
    await run_script(agent, SCRIPT_ERR, tr)

    tr.write("\n## Session isolation check")
    for sid in (ALICE, BOB):
        st = await agent.get_state(sid)
        tr.write(f"  {sid}: messages={len(st.get('messages', []))} mode={st.get('mode')} "
                 f"lead_status={st.get('lead_status')} lead_id={st.get('lead_id')} "
                 f"active_products={st.get('active_products')}")
    await agent.aclose()

    tr.write("\n## 4. Process restart — new agent instance, same SQLite checkpoint store")
    agent = await SalesAgent.create(s)
    st = await agent.get_state(ALICE)
    tr.write(f"  restored {ALICE}: {len(st.get('messages', []))} messages, lead_id={st.get('lead_id')}, "
             f"active_products={st.get('active_products')}, summary={'yes' if st.get('summary') else 'no'}")
    await run_script(agent, SCRIPT_RESTART, tr)

    tr.write("\n## 5. Leads stored in SQLite, read back via the MCP tool `list_leads`")
    out = await agent.gateway.list_leads(limit=10)
    for row in out.get("leads", []):
        tr.write("  " + json.dumps({k: row[k] for k in ("lead_id", "session_id", "full_name", "occupation",
                                                        "monthly_income", "income_text", "phone", "product_interest",
                                                        "consent", "created_at")}, ensure_ascii=False))
    await agent.aclose()
    tr.write(f"\n# transcript: {tr.path.relative_to(ROOT)} | structured turn log: logs/turns.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
