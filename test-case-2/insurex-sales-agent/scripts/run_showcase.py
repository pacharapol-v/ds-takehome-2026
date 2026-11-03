"""Showcase demo: one realistic customer journey plus a short second customer.

    python scripts/run_showcase.py

Somchai (Thai, casual) asks about car insurance (rejected: not in the knowledge base),
browses the catalogue, dives into ตลอดชีพ 90/20 with follow-ups, compares it with 99/99,
shows interest and gives his details in one message, then consents - the lead is saved
through the MCP tool. John (English) is interleaved: his "it" must resolve to *his* plan.

Runs on a temporary state folder, so existing sessions and leads are not touched.
Output: logs/showcase_<llm>.log
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_demo import Transcript  # noqa: E402

from insurex_agent.agent.app import SalesAgent  # noqa: E402
from insurex_agent.config import get_settings  # noqa: E402

SOMCHAI = "somchai-th"
JOHN = "john-en"

SCRIPT = [
    (SOMCHAI, "ประกันรถมีมั้ย"),
    (SOMCHAI, "แล้วมีประกันอะไรบ้าง"),
    (SOMCHAI, "90/20 คืออะไร"),
    (JOHN, "hi, do you have term life insurance?"),
    (SOMCHAI, "ผมอายุ 35 ต้องจ่ายปีละเท่าไหร่"),
    (JOHN, "how long does it cover?"),
    (SOMCHAI, "จ่ายรายเดือนได้มั้ย"),
    (SOMCHAI, "แล้วถ้าเทียบกับ 99/99 ล่ะ ต่างกันยังไง"),
    (SOMCHAI, "โอเค สนใจ 90/20"),
    (SOMCHAI, "สมชาย ใจดี วิศวกร 50000บาท 0812345678"),
    (SOMCHAI, "ยืนยัน"),
]


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    base = get_settings()
    s = base.model_copy(update={"state_dir": Path(tempfile.mkdtemp(prefix="insurex_showcase_"))})
    s.ensure_dirs()
    tag = "offline-rules" if s.llm_provider == "offline" else f"{s.llm_provider}-{s.llm_model}"
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    tr = Transcript(logs / f"showcase_{tag.replace('/', '_').replace(':', '_')}.log")
    tr.write(f"# InsureX Sales Agent showcase - {datetime.now().isoformat(timespec='seconds')}")
    tr.write(f"# LLM: {s.llm_provider}:{s.llm_model} | embeddings: {s.embedding_provider}:{s.embedding_model} | "
             f"leads backend: {s.leads_backend} | fresh state folder")

    agent = await SalesAgent.create(s, turn_log=None)
    welcomed: set[str] = set()
    for sid, msg in SCRIPT:
        if sid not in welcomed:                       # what the CLI shows when a session starts
            welcomed.add(sid)
            tr.write(f"\n[{sid}] (session start) BOT : " +
                     (await agent.welcome(sid)).replace("\n", "\n" + " " * (len(sid) + 26)))
        tr.turn(await agent.chat(sid, msg))

    tr.write("\n## Session isolation check")
    for sid in (SOMCHAI, JOHN):
        st = await agent.get_state(sid)
        tr.write(f"  {sid}: messages={len(st.get('messages', []))} mode={st.get('mode')} "
                 f"lead_status={st.get('lead_status')} lead_id={st.get('lead_id')} "
                 f"active_products={st.get('active_products')}")

    tr.write("\n## Leads stored in SQLite, read back via the MCP tool `list_leads`")
    out = await agent.gateway.list_leads(limit=10)
    for row in out.get("leads", []):
        tr.write("  " + json.dumps({k: row[k] for k in ("lead_id", "session_id", "full_name", "occupation",
                                                        "monthly_income", "income_text", "phone", "product_interest",
                                                        "consent", "created_at")}, ensure_ascii=False))
    await agent.aclose()
    tr.write(f"\n# transcript: {tr.path.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
