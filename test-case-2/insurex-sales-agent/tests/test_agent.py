"""Workflow tests for the LangGraph agent (offline: RuleBrain, local embeddings)."""

import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from insurex_agent.agent.app import SalesAgent
from insurex_agent.agent.brain import LLMBrain, RuleBrain
from insurex_agent.agent.schemas import GeneratedAnswer, GradeResult
from insurex_agent.ingestion.catalog import PRODUCTS, unknown_plan_codes
from insurex_agent.leads.gateway import DirectLeadsGateway
from insurex_agent.leads.repository import LeadRepository
from insurex_agent.memory.sessions import SessionRegistry


def run(coro):
    return asyncio.run(coro)


async def _agent(settings, retriever, **kw):
    kw.setdefault("checkpointer", InMemorySaver())
    kw.setdefault("gateway", DirectLeadsGateway(settings))
    return await SalesAgent.create(settings, retriever=retriever, turn_log=None, **kw)


LEAD_SCRIPT = [
    "ตลอดชีพ 90/20 ผู้หญิงอายุ 35 ทุน 500,000 จ่ายเบี้ยปีละเท่าไหร่",
    "สนใจครับ ให้ตัวแทนติดต่อกลับได้",
    "ผมชื่อ สมชาย ใจดี อาชีพ วิศวกร",
    "รายได้ 5 หมื่นต่อเดือน เบอร์ 081-234-5678",
    "ยืนยันครับ",
]


def test_rag_turn_runs_full_pipeline(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever)
        return await a.chat("s1", LEAD_SCRIPT[0])

    r = run(go())
    assert r.trace[:3] == ["prepare_turn", "route", "retrieve#1"]
    assert "verify_answer:ok" in r.trace and "finalize_answer" in r.trace
    assert "10,650" in r.answer                         # correct figure from the brochure
    assert r.sources and "whole_life_90_20.pdf" in r.sources[0]
    assert r.active_products == ["whole_life_90_20"]


def test_unknown_topic_goes_through_corrective_loop_then_fallback(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever)
        return await a.chat("s1", "ประกันรถยนต์ชั้น 1 ราคาเท่าไหร่")

    r = run(go())
    assert "rewrite_query" in r.trace and "retrieve#2" in r.trace      # cycle 1
    assert r.trace[-2] == "no_answer:not_found"
    assert "1314" in r.answer                                          # graceful hand-off


def test_lead_collection_end_to_end_persists_validated_record(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever)
        return [await a.chat("s1", m) for m in LEAD_SCRIPT]

    turns = run(go())
    assert turns[1].mode == "lead_collection" and turns[1].lead_status == "collecting"
    assert turns[3].lead_status == "awaiting_confirmation"            # explicit consent step before saving
    final = turns[-1]
    assert final.lead_status == "saved" and final.lead_id.startswith("LD-")
    row = LeadRepository(fresh_state.leads_db_path).get(final.lead_id)
    assert row["full_name"] == "สมชาย ใจดี" and row["occupation"] == "วิศวกร"
    assert row["monthly_income"] == 50000 and row["phone"] == "0812345678"
    assert row["product_interest"] == "whole_life_90_20" and row["consent"] == 1 and row["session_id"] == "s1"


def test_question_during_lead_collection_is_answered_and_collection_resumes(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever)
        await a.chat("s1", "สนใจ ตลอดชีพ 99/99 ครับ")
        return await a.chat("s1", "ผมชื่อ สมชาย ใจดี แล้ว 99/99 รับประกันอายุสูงสุดกี่ปี")

    r = run(go())
    assert "retrieve#1" in r.trace and "extract_lead" in r.trace
    assert "75 ปี" in r.answer                                         # answered the detour question
    assert "รบกวนขอ" in r.answer and r.lead["full_name"] == "สมชาย ใจดี"   # and kept collecting


def test_cancel_discards_partial_lead(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever)
        await a.chat("s1", "สนใจครับ")
        await a.chat("s1", "ผมชื่อ สมชาย ใจดี")
        return await a.chat("s1", "ยกเลิกครับ ไม่สะดวกให้ข้อมูล")

    r = run(go())
    assert r.lead_status == "cancelled" and r.lead == {} and r.mode == "qa"
    assert LeadRepository(fresh_state.leads_db_path).list() == []


def test_sessions_are_isolated(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever)
        # run two customers concurrently
        await asyncio.gather(a.chat("alice", "สนใจคุ้มมั่นใจชัวร์ครับ"), a.chat("bob", "ตลอดชีพ 99/99 รับอายุถึงกี่ปี"))
        await a.chat("alice", "ผมชื่อ สมชาย ใจดี อาชีพ วิศวกร")
        return await a.get_state("alice"), await a.get_state("bob")

    alice, bob = run(go())
    assert alice["mode"] == "lead_collection" and alice["lead"]["full_name"] == "สมชาย ใจดี"
    assert bob["mode"] == "qa" and not bob.get("lead")
    assert alice["active_products"] == ["khum_manjai_sure_term"] and bob["active_products"] == ["whole_life_99_99"]
    assert all("99/99" not in str(m.content) for m in alice["messages"])


def test_session_survives_restart_with_sqlite_checkpointer(fresh_state, retriever):
    async def first():
        a = await SalesAgent.create(fresh_state, retriever=retriever, gateway=DirectLeadsGateway(fresh_state),
                                    turn_log=None)                    # default = AsyncSqliteSaver on disk
        await a.chat("persist", "สนใจตลอดชีพ 90/20 ครับ")
        await a.chat("persist", "ผมชื่อ สมชาย ใจดี")
        await a.aclose()

    async def second():
        a = await SalesAgent.create(fresh_state, retriever=retriever, gateway=DirectLeadsGateway(fresh_state),
                                    turn_log=None)
        st = await a.get_state("persist")
        r = await a.chat("persist", "อาชีพ วิศวกร")
        await a.aclose()
        return st, r

    run(first())
    st, r = run(second())
    assert len(st["messages"]) == 4 and st["lead"]["full_name"] == "สมชาย ใจดี"
    assert r.lead["occupation"] == "วิศวกร" and r.lead["full_name"] == "สมชาย ใจดี"


def test_catalogue_question_is_answered_from_the_catalogue_not_rag(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever)
        return await a.chat("s1", "มีประกันอะไรบ้าง"), await a.chat("s2", "คุ้มชีวามีแบบอะไรบ้าง")

    catalog, specific = run(go())
    assert catalog.trace == ["prepare_turn", "route", "catalog", "finalize"]
    assert all(p.name_th in catalog.answer for p in PRODUCTS)
    assert "retrieve#1" in specific.trace                     # one named product -> normal RAG question


class _CatalogHappyBrain(RuleBrain):
    """Router that labels everything 'catalog', as the LLM sometimes did for "ประกันรถยนต์มีอะไรบ้าง"."""

    async def route(self, message, history, state):
        d = await super().route(message, history, state)
        d.intent = "catalog"
        return d


def test_other_insurance_type_is_never_answered_with_our_catalogue(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever, brain=_CatalogHappyBrain())
        return await a.chat("s1", "ประกันรถยนต์มีอะไรบ้าง"), await a.chat("s2", "มีประกันอะไรบ้างที่สามารถลดหย่อนภาษีได้")

    car, tax = run(go())
    assert "catalog" not in car.trace and car.trace[-2] == "no_answer:not_found"
    assert "catalog" in tax.trace                             # "สามารถ" contains "รถ" but is not car insurance


def test_unknown_plan_code_is_not_guessed_and_starts_no_lead(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever)
        return await a.chat("s1", "สนใจคุ้มชีวา 99/50"), await a.chat("s2", "สนใจคุ้มชีวา 90/10")

    unknown, known = run(go())
    assert unknown.trace[2] == "unknown_plan:99/50" and "99/50" in unknown.answer
    assert unknown.mode == "qa" and unknown.lead == {} and unknown.lead_status == "none"
    assert known.mode == "lead_collection" and known.lead["product_interest"] == "khum_cheeva_90_10"
    assert unknown_plan_codes("ตลอดชีพ 90/20 กับ 99/50 นัดวันที่ 12/11/2026") == ["99/50"]   # dates are not plans


def test_welcome_only_for_a_session_without_history(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever)
        first = await a.welcome("s1")
        await a.chat("s1", "สวัสดีครับ")
        return first, await a.welcome("s1")

    first, resumed = run(go())
    assert "InsureX Assistant" in first and "ตลอดชีพ 90/20" in first and "สนใจ" in first
    assert resumed is None


def test_forget_erases_conversation_leads_and_registry_of_one_session(fresh_state, retriever):
    async def go():
        a = await SalesAgent.create(fresh_state, retriever=retriever, gateway=DirectLeadsGateway(fresh_state),
                                    turn_log=None)                    # on-disk SQLite checkpointer
        saved = [await a.chat("s1", m) for m in LEAD_SCRIPT][-1]
        await a.chat("s2", "สนใจตลอดชีพ 90/20 ครับ")
        out = await a.forget("s1")
        s1, s2 = await a.get_state("s1"), await a.get_state("s2")
        await a.aclose()
        return saved, out, s1, s2

    saved, out, s1, s2 = run(go())
    assert out == {"session_id": "s1", "leads_deleted": [saved.lead_id], "session_existed": True}
    assert s1 == {}                                                     # conversation gone
    assert s2["lead_status"] == "collecting"                            # other session untouched
    assert LeadRepository(fresh_state.leads_db_path).list(session_id="s1") == []
    registry = SessionRegistry(fresh_state.sessions_db_path)
    assert registry.get("s1") is None and registry.get("s2") is not None


def test_long_history_is_summarised_and_trimmed(fresh_state, retriever):
    s = fresh_state.model_copy(update={"summarize_after": 6, "history_window": 4})

    async def go():
        a = await _agent(s, retriever)
        for q in ["สวัสดีครับ", "คุ้มชีวา 90/10 จ่ายเบี้ยกี่ปี", "ตลอดชีพ 99/99 รับอายุถึงกี่ปี", "ขอบคุณครับ"]:
            r = await a.chat("long", q)
        return r, await a.get_state("long")

    r, st = run(go())
    assert "summarize_history" in r.trace
    assert len(st["messages"]) <= 6 and st["summary"]


class _BrokenModel:
    """Simulates an LLM outage (every call raises)."""

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, *_a, **_k):
        raise ConnectionError("503 upstream unavailable")


def test_llm_outage_degrades_to_rules_instead_of_failing(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever, brain=LLMBrain(_BrokenModel(), fresh_state))
        return await a.chat("s1", "ตลอดชีพ 99/99 ผู้หญิงอายุ 40 เบี้ยปีละเท่าไหร่")

    r = run(go())
    assert {"route", "grade", "generate"} <= set(r.degraded_steps)
    assert "4,830" in r.answer


class _HallucinatingOnceBrain(RuleBrain):
    calls = 0

    async def generate(self, question, parents, history, language, feedback=""):
        type(self).calls += 1
        if type(self).calls == 1:
            return GeneratedAnswer(answer="เบี้ยประกันปีละ 99,999 บาทครับ [1]", cited_blocks=[1], fully_answered=True)
        return await super().generate(question, parents, history, language, feedback)


def test_unsupported_number_triggers_regeneration(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever, brain=_HallucinatingOnceBrain())
        return await a.chat("s1", "ตลอดชีพ 99/99 ผู้หญิงอายุ 40 เบี้ยปีละเท่าไหร่")

    r = run(go())
    assert r.trace.count("verify_answer:fail") == 1 and "generate#2" in r.trace      # cycle 2
    assert "99,999" not in r.answer


class _NeverAnswerableBrain(RuleBrain):
    """Grader that finds blocks relevant but never 'fully answerable' (as an LLM does for 'ต่างกันยังไง')."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    async def grade(self, question, parents):
        return GradeResult(relevant_blocks=self.blocks, answerable=False)


def test_partial_evidence_is_answered_on_last_attempt_instead_of_refused(fresh_state, retriever):
    async def go(blocks):
        a = await _agent(fresh_state, retriever, brain=_NeverAnswerableBrain(blocks))
        return await a.chat("s1", "ตลอดชีพ 99/99 ผู้หญิงอายุ 40 เบี้ยปีละเท่าไหร่")

    partial, nothing = run(go([1, 2])), run(go([]))
    t = partial.trace
    assert t[2:6] == ["retrieve#1", "grade_documents:0/4", "rewrite_query", "retrieve#2"]   # cycle 1 still runs
    assert any(s.startswith("grade_documents:partial 2/") for s in t) and "generate#1" in t
    assert not any(s.startswith("no_answer") for s in t)
    assert nothing.trace[-2] == "no_answer:not_found"           # nothing relevant -> still a refusal


class _FlakyGateway(DirectLeadsGateway):
    failures = 1

    async def save_lead(self, record):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("MCP server crashed")
        return await super().save_lead(record)


def test_tool_failure_keeps_lead_and_allows_retry(fresh_state, retriever):
    async def go():
        a = await _agent(fresh_state, retriever, gateway=_FlakyGateway(fresh_state))
        turns = [await a.chat("s1", m) for m in LEAD_SCRIPT[1:]]
        retry = await a.chat("s1", "ยืนยัน")
        return turns[-1], retry

    failed, retry = run(go())
    assert failed.lead_status == "awaiting_confirmation" and "tool_error:ConnectionError" in failed.trace
    assert retry.lead_status == "saved" and retry.lead_id


class _ScriptedModel:
    """Stands in for a real chat model: returns schema-valid outputs and records the prompts,
    so the LLM path (prompt formatting, structured outputs, citations) is exercised offline."""

    def __init__(self):
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        model = self

        class _Runner:
            async def ainvoke(self, messages):
                text = "\n".join(str(m.content) for m in messages)
                model.prompts.append(text)
                name = schema.__name__
                if name == "RouteDecision":
                    return schema(intent="product_question", wants_contact=False, has_question=True,
                                  products=["whole_life_99_99", "not_a_product"],
                                  standalone_query="ตลอดชีพ 99/99 เพศหญิง อายุ 40 เบี้ยประกันภัยรายปี", language="th")
                if name == "GradeResult":
                    return schema(relevant_blocks=[1, 99], answerable=True)
                if name == "GeneratedAnswer":
                    return schema(answer="ตัวอย่างเบี้ยรายปีสำหรับเพศหญิงอายุ 40 ปี ทุน 300,000 บาท คือ 4,830 บาทครับ [1]",
                                  cited_blocks=[1], fully_answered=True)
                if name == "GroundingCheck":
                    return schema(grounded=True, unsupported_claims=[])
                raise AssertionError(name)

        return _Runner()

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage("- สรุป")


def test_llm_path_prompts_and_structured_outputs(fresh_state, retriever):
    model = _ScriptedModel()

    async def go():
        a = await _agent(fresh_state, retriever, brain=LLMBrain(model, fresh_state))
        return await a.chat("s1", "99/99 ผู้หญิง 40 จ่ายปีละเท่าไหร่")

    r = run(go())
    assert not r.degraded_steps
    assert r.trace == ["prepare_turn", "route", "retrieve#1", "grade_documents:1/4", "generate#1",
                       "verify_answer:ok", "finalize_answer", "finalize"]
    assert "4,830" in r.answer and "whole_life_99_99.pdf" in r.answer          # citation appended
    router_prompt, grader_prompt, gen_prompt = model.prompts[0], model.prompts[1], model.prompts[2]
    assert "whole_life_90_20" in router_prompt                                   # catalogue in router prompt
    assert "[1] ตลอดชีพ 99/99" in grader_prompt                                   # numbered context blocks
    assert "ONLY the numbered CONTEXT" in gen_prompt and "Thai" in gen_prompt


def test_llm_number_check_catches_invented_premium(fresh_state, retriever):
    class _Liar(_ScriptedModel):
        def with_structured_output(self, schema):
            inner = super().with_structured_output(schema)
            if schema.__name__ != "GeneratedAnswer":
                return inner

            class _R:
                async def ainvoke(self, messages):
                    return schema(answer="เบี้ยประมาณ 5,100 บาทครับ [1]", cited_blocks=[1], fully_answered=True)
            return _R()

    async def go():
        a = await _agent(fresh_state, retriever, brain=LLMBrain(_Liar(), fresh_state))
        return await a.chat("s1", "99/99 ผู้หญิง 40 จ่ายปีละเท่าไหร่")

    r = run(go())
    assert r.trace.count("verify_answer:fail") == 2 and r.trace[-2] == "no_answer:unverified"
    assert "5,100" not in r.answer
