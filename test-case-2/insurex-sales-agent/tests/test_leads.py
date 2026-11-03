import asyncio

import pytest
from pydantic import ValidationError

from insurex_agent.leads.models import (LeadExtraction, LeadRecord, ground_extraction, missing_fields,
                                        normalize_phone)
from insurex_agent.leads.repository import LeadRepository


@pytest.mark.parametrize("raw,expected", [
    ("081-234-5678", "0812345678"), ("+66 81 234 5678", "0812345678"), ("๐๘๑๒๓๔๕๖๗๘", "0812345678"),
    ("02 123 4567", "021234567"), ("12345", None), ("0112345678", None), (None, None),
])
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def _record(**kw):
    base = dict(session_id="s1", full_name="คุณสมชาย ใจดี", occupation="วิศวกร", monthly_income=50000,
                phone="081-234-5678", consent=True)
    base.update(kw)
    return LeadRecord(**base)


def test_lead_record_normalises_and_validates():
    r = _record()
    assert r.full_name == "สมชาย ใจดี" and r.phone == "0812345678"
    with pytest.raises(ValidationError):
        _record(consent=False)                       # PDPA: no consent, no record
    with pytest.raises(ValidationError):
        _record(phone="123")
    with pytest.raises(ValidationError):
        _record(monthly_income=0)
    with pytest.raises(ValidationError):
        _record(full_name="0812345678")


def test_grounding_rejects_values_not_in_message():
    msg = "ผมชื่อสมชาย ใจดี ทำงานเป็นวิศวกร"
    ext = LeadExtraction(full_name="สมชาย ใจดี", occupation="วิศวกร", phone="0899999999",   # hallucinated phone
                         income_amount=80000, income_text="8 หมื่น")                      # hallucinated income
    clean, rejected = ground_extraction(ext, msg)
    assert clean == {"full_name": "สมชาย ใจดี", "occupation": "วิศวกร"}
    assert set(rejected) == {"phone", "monthly_income"}


def test_grounding_income_normalised_to_monthly():
    clean, _ = ground_extraction(LeadExtraction(income_text="6 แสนต่อปี", income_amount=600000,
                                                income_period="yearly"), "รายได้ 6 แสนต่อปีครับ")
    assert clean["monthly_income"] == 50000
    clean, _ = ground_extraction(LeadExtraction(), "เบอร์ 081 234 5678 ครับ")      # model missed the number
    assert clean["phone"] == "0812345678"


def test_missing_fields():
    assert missing_fields({"full_name": "a", "phone": "0812345678"}) == ["occupation", "monthly_income"]


def test_repository_upsert_is_idempotent(tmp_path):
    repo = LeadRepository(tmp_path / "leads.sqlite")
    a = repo.upsert(_record())
    b = repo.upsert(_record(occupation="ผู้จัดการ"))            # retry / correction -> same lead
    assert a["status"] == "created" and b["status"] == "updated" and a["lead_id"] == b["lead_id"]
    rows = repo.list()
    assert len(rows) == 1 and rows[0]["occupation"] == "ผู้จัดการ"
    assert [e["event"] for e in repo.events(a["lead_id"])] == ["created", "updated"]   # audit trail


def test_repository_delete_by_session_erases_leads_and_events(tmp_path):
    repo = LeadRepository(tmp_path / "leads.sqlite")
    gone = repo.upsert(_record())
    kept = repo.upsert(_record(session_id="other"))
    assert repo.delete_by_session(_record().session_id) == [gone["lead_id"]]
    assert repo.get(gone["lead_id"]) is None and repo.events(gone["lead_id"]) == []   # PII copy in events gone too
    assert [r["lead_id"] for r in repo.list()] == [kept["lead_id"]]                    # other sessions untouched
    assert repo.delete_by_session("no-such-session") == []


def test_mcp_server_roundtrip(tmp_path):
    """Spawns the real MCP server over stdio and calls its tools."""
    from insurex_agent.config import Settings
    from insurex_agent.leads.gateway import MCPLeadsGateway

    s = Settings(state_dir=tmp_path, _env_file=None)

    async def run():
        gw = MCPLeadsGateway(s)
        try:
            assert await gw.tool_names() == ["delete_leads", "get_lead", "list_leads", "save_lead"]
            ok = await gw.save_lead(_record())
            bad = await gw._call("save_lead", {"session_id": "s", "full_name": "x", "occupation": "y",
                                               "monthly_income": -1, "phone": "1", "consent": False})
            listed = await gw.list_leads()
            deleted = await gw.delete_leads(_record().session_id)
            after = await gw.list_leads()
            return ok, bad, listed, deleted, after
        finally:
            await gw.aclose()

    ok, bad, listed, deleted, after = asyncio.run(run())
    assert ok["ok"] and ok["lead_id"].startswith("LD-")
    assert not bad["ok"] and any("phone" in e for e in bad["errors"])
    assert listed["leads"][0]["phone"] == "0812345678"
    assert deleted == {"ok": True, "deleted": [ok["lead_id"]]} and after["leads"] == []
