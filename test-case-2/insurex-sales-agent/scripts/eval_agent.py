"""End-to-end answer evaluation: every question in eval/qa_set.jsonl goes through the
full LangGraph agent (fresh session each), and the final answer is scored.

    python scripts/eval_agent.py [--limit N]

Scoring
  answerable    PASS if the answer contains the gold fact strings (commas/spaces ignored)
  negative      PASS if the agent took the `no_answer` fallback (did not make something up)
  not_in_doc    PASS if the agent did not quote any amount that is absent from the brochure
                AND either fell back or said the brochure does not specify it
Writes eval/results/agent_<llm>.md and .jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from insurex_agent.agent.app import SalesAgent  # noqa: E402
from insurex_agent.config import get_settings  # noqa: E402
from insurex_agent.leads.gateway import DirectLeadsGateway  # noqa: E402

NOT_SPECIFIED = ("ไม่ได้ระบุ", "ไม่พบ", "ไม่มีข้อมูล", "not specified", "not available", "couldn't find", "does not")


def norm(s: str) -> str:
    # lower(): English answers ("Dividends") match English gold facts; en/em dashes count as "-" ("20–50 ปี")
    return re.sub(r"[\s,]", "", re.sub(r"[–—−]", "-", s)).lower()


def score(q: dict, res, kb_numbers: set[str]) -> bool:
    ans = norm(res.answer)
    if q["answerable"]:
        ok_all = all(norm(f) in ans for f in q["facts_all"])
        ok_any = not q["facts_any"] or any(norm(f) in ans for f in q["facts_any"])
        return ok_all and ok_any and not any(t.startswith("no_answer") for t in res.trace)
    fell_back = any(t.startswith("no_answer") for t in res.trace)
    if q["category"] == "negative":
        return fell_back or res.intent == "out_of_scope"
    amounts = {n.replace(",", "") for n in re.findall(r"\d{1,3}(?:,\d{3})+|\d{4,}", res.answer)}
    invented = amounts - kb_numbers
    return not invented and (fell_back or any(p in res.answer for p in NOT_SPECIFIED))


async def main(limit: int | None) -> None:
    s = get_settings()
    qs = [json.loads(l) for l in (ROOT / "eval" / "qa_set.jsonl").read_text(encoding="utf-8").splitlines() if l]
    qs = qs[:limit] if limit else qs
    agent = await SalesAgent.create(s, checkpointer=InMemorySaver(), gateway=DirectLeadsGateway(s), turn_log=None)
    kb_text = " ".join(c.text for c in agent.retriever.kb.children)
    kb_numbers = {n.replace(",", "") for n in re.findall(r"\d{1,3}(?:,\d{3})+|\d{4,}", kb_text)}
    rows = []
    t0 = time.time()
    for i, q in enumerate(qs):
        res = await agent.chat(f"eval-{i}", q["question"])
        ok = score(q, res, kb_numbers)
        rows.append({"id": q["id"], "category": q["category"], "pass": ok, "question": q["question"],
                     "answer": res.answer, "trace": res.trace, "latency_ms": res.latency_ms,
                     "degraded": res.degraded_steps})
        print(f"[{'PASS' if ok else 'FAIL'}] {q['id']:<22} {res.latency_ms:>6} ms  {' → '.join(res.trace[2:-1])}")
    await agent.aclose()

    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["pass"])
    total = sum(r["pass"] for r in rows)
    lat = sorted(r["latency_ms"] for r in rows)
    retries = sum(any(t == "rewrite_query" for t in r["trace"]) for r in rows)
    regens = sum(any(t.startswith("generate#2") for t in r["trace"]) for r in rows)
    tag = f"{s.llm_provider}_{s.llm_model}".replace("/", "_").replace(":", "_")
    md = [f"# Agent evaluation — LLM `{s.llm_provider}:{s.llm_model}`, embeddings `{s.embedding_provider}:{s.embedding_model}`",
          "", f"**{total}/{len(rows)} passed** ({total / len(rows):.0%}) in {time.time() - t0:.0f}s; "
          f"median latency {lat[len(lat) // 2]} ms; corrective retrievals: {retries}; regenerations: {regens}", "",
          "| Category | Pass | Total |", "|---|---|---|"]
    md += [f"| {c} | {sum(v)} | {len(v)} |" for c, v in sorted(by_cat.items())]
    md += ["", "Failures:", ""] + [f"- `{r['id']}`: {r['question']}" for r in rows if not r["pass"]]
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"agent_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out / f"agent_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    asyncio.run(main(ap.parse_args().limit))
