"""LangGraph workflow for the InsureX sales assistant.

                       ┌──────────────┐
             START ──▶ │ prepare_turn │  reset per-turn fields, summarise long history
                       └──────┬───────┘
                              ▼
                       ┌──────────────┐   intent + standalone query + product resolution
                       │    route     │
                       └──────┬───────┘
     ┌──────────┬─────────────┼──────────────┬──────────────┬─────────────┐
     ▼          ▼             ▼              ▼              ▼             ▼
 save_lead  cancel_lead  extract_lead     retrieve ◀──┐  smalltalk   out_of_scope
  (MCP)                       │              │        │  catalog, unknown_plan (templated, deterministic)
                              ▼              ▼        │ (cycle 1: corrective retrieval)
                       has question? ─▶ grade_documents ──▶ rewrite_query
                              │              │ relevant
                              ▼              ▼
                       lead_next_step    generate ◀──┐
                                             │       │ (cycle 2: self-correction)
                                             ▼       │
                                       verify_answer ┘──▶ no_answer (fallback)
                                             │ grounded
                                             ▼
                                       finalize_answer ──▶ finalize ──▶ END

All language-understanding steps go through `brain` (LLM or rule fallback);
everything that touches data integrity (lead validation, consent, persistence,
citations, number checking) is deterministic code.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from ..config import Settings, get_settings
from ..ingestion.catalog import (OTHER_INSURANCE_RE, PRODUCT_BY_ID, PRODUCTS, catalog_overview_en,
                                 catalog_overview_th, unknown_plan_codes)
from ..leads.gateway import LeadsGateway
from ..leads.models import (FIELD_LABEL_EN, FIELD_LABEL_TH, LeadRecord, format_lead_summary, ground_extraction,
                            missing_fields)
from ..retrieval.hybrid import HybridRetriever, RetrievedParent, detect_products
from .brain import _CONFIRM_RE, LLMBrain, detect_language, format_context, unsupported_numbers

log = logging.getLogger(__name__)


def welcome_message() -> str:
    """Opening message for a new session: who the agent is and how it can help.
    Deterministic (built from the product catalogue), so it is instant and cannot drift."""
    products = "\n".join(f"  • {p.name_th}" for p in PRODUCTS)
    return ("สวัสดีครับ ผมคือ InsureX Assistant ผู้ช่วย AI ด้านประกันชีวิตของ InsureX\n\n"
            "ผมช่วยคุณได้ดังนี้ครับ\n"
            f"1. ตอบคำถามเกี่ยวกับแผนประกันชีวิต {len(PRODUCTS)} แบบ เช่น ความคุ้มครอง เงื่อนไขการรับประกัน "
            "และตัวอย่างเบี้ย พร้อมอ้างอิงหน้าโบรชัวร์\n"
            f"{products}\n"
            "2. เปรียบเทียบแผนประกันให้เห็นความแตกต่าง\n"
            "3. ประสานให้ตัวแทนติดต่อกลับ เพียงพิมพ์ 'สนใจ'\n\n"
            "ตัวอย่างคำถาม: \"ตลอดชีพ 90/20 ผู้หญิงอายุ 35 ทุน 500,000 จ่ายปีละเท่าไหร่\"\n\n"
            "วันนี้ให้ผมช่วยเรื่องไหนดีครับ\n"
            "(You can also ask in English.)")


def _trace_reducer(old: list[str] | None, new: list[str] | None) -> list[str]:
    """Append node names; a list starting with '__reset__' starts a new turn."""
    if new and new[0] == "__reset__":
        return list(new[1:])
    return (old or []) + (new or [])


class AgentState(TypedDict, total=False):
    # --- persisted across turns (per session / thread) ---
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    summary: str
    language: str
    mode: str                     # "qa" | "lead_collection"
    lead: dict                    # partial lead being collected
    lead_status: str              # none | collecting | awaiting_confirmation | saved | cancelled
    lead_id: str | None
    lead_rejected: list[str]
    active_products: list[str]    # products in focus (for follow-up questions)
    # --- per turn ---
    route: dict
    query: str
    retrieval_attempt: int
    generation_attempt: int
    retrieved: list[dict]
    relevant: list[dict]
    draft: dict
    verification: dict
    answer: str
    sources: list[str]
    turn_trace: Annotated[list[str], _trace_reducer]


def _serialize(p: RetrievedParent) -> dict:
    return {"parent_id": p.parent_id, "text": p.text, "metadata": p.metadata, "score": round(p.score, 5),
            "citation": p.citation(),
            "hits": [{"text": h.text, "score": round(h.score, 5), "chunk_type": h.metadata.get("chunk_type")}
                     for h in p.hits]}


def _last_human(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _last_ai(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return str(m.content)
    return ""


def _product_name(pid: str | None) -> str | None:
    return PRODUCT_BY_ID[pid].name_th if pid in PRODUCT_BY_ID else None


class SalesAgentGraph:
    def __init__(self, brain, retriever: HybridRetriever, gateway: LeadsGateway, settings: Settings | None = None):
        self.brain = brain
        self.retriever = retriever
        self.gateway = gateway
        self.s = settings or get_settings()

    # ------------------------------------------------------------------ turn setup
    async def prepare_turn(self, state: AgentState) -> dict:
        msgs = state.get("messages", [])
        update: dict[str, Any] = {
            "route": {}, "query": "", "retrieval_attempt": 0, "generation_attempt": 0, "retrieved": [],
            "relevant": [], "draft": {}, "verification": {}, "answer": "", "sources": [],
            "turn_trace": ["__reset__", "prepare_turn"],
            "language": detect_language(_last_human(state)),
            "mode": state.get("mode") or "qa",
            "lead": state.get("lead") or {},
            "lead_status": state.get("lead_status") or "none",
            "active_products": state.get("active_products") or [],
        }
        # memory management: fold old messages into a running summary
        if len(msgs) > self.s.summarize_after:
            old = msgs[: -self.s.history_window]
            update["summary"] = await self.brain.summarize(state.get("summary", ""), old)
            update["messages"] = [RemoveMessage(id=m.id) for m in old if m.id]
            update["turn_trace"].append("summarize_history")
        return update

    # ------------------------------------------------------------------ routing
    async def route(self, state: AgentState) -> dict:
        message = _last_human(state)
        history = state.get("messages", [])[:-1]
        d = await self.brain.route(message, history, state)
        lead_status = state.get("lead_status", "none")
        if lead_status == "awaiting_confirmation" and _CONFIRM_RE.search(message):
            d.intent, d.has_question = "confirm", False      # deterministic: a bare "yes" always confirms
        if d.intent == "confirm" and lead_status != "awaiting_confirmation":
            d.intent = "smalltalk" if not d.has_question else "product_question"

        # language: the script detector is deterministic and reliable; the router's guess (which can use history)
        # only decides for messages too short to tell, e.g. "ok" (the LLM tends to answer "th" for English)
        if sum(c.isalpha() for c in message) >= 4:
            d.language = detect_language(message)
        # product guard: when the customer names products, the router may not add others (except ones already
        # in focus). It once turned "คุ้มชีวาแต่ละแบบ" into a comparison with คุ้มชีวี เอ็กซ์ตร้า.
        named = detect_products(message)
        if named:
            allowed = set(named) | set(state.get("active_products") or [])
            kept = [p for p in d.products if p in allowed]
            if kept != d.products:
                log.info("router products %s -> %s (not named by the customer)", d.products, kept or named)
                d.products = kept or named
                d.standalone_query = message      # the rewrite was built around the added product
        # not catalogue questions: one named product ("คุ้มชีวามีแบบอะไรบ้าง") or an insurance type we do not sell
        # ("ประกันรถยนต์มีอะไรบ้าง" - the catalogue reply would read as if our life plans were car insurance)
        if d.intent == "catalog" and (named or OTHER_INSURANCE_RE.search(message)):
            d.intent, d.has_question = "product_question", True
        # plan codes that do not exist ("คุ้มชีวา 99/50"): never guess a product or start a lead for them
        unknown = unknown_plan_codes(message)
        if unknown:
            log.info("unknown plan code(s) %s - asking the customer which plan they mean", unknown)
            d.wants_contact, d.products = False, []
        update: dict[str, Any] = {"route": {**d.model_dump(), "unknown_plan_codes": unknown},
                                  "language": d.language, "query": d.standalone_query, "turn_trace": ["route"]}
        if d.products:
            update["active_products"] = d.products
        if d.wants_contact:
            lead = dict(state.get("lead") or {})
            focus = d.products or state.get("active_products") or []
            if focus:
                lead["product_interest"] = focus[0]
            update.update({"mode": "lead_collection", "lead": lead})
            if lead_status in ("none", "saved", "cancelled"):
                update["lead_status"] = "collecting"
        return update

    def after_route(self, state: AgentState) -> str:
        r, mode, status = state["route"], state.get("mode"), state.get("lead_status")
        if r["intent"] == "confirm" and status == "awaiting_confirmation":
            return "save_lead"
        if r["intent"] == "cancel" and mode == "lead_collection":
            return "cancel_lead"
        # deterministic replies; during lead collection `finalize` then asks for the remaining fields
        if r.get("unknown_plan_codes"):
            return "unknown_plan"
        if r["intent"] == "catalog":
            return "catalog"
        if mode == "lead_collection":
            return "extract_lead"
        if r["intent"] == "out_of_scope":
            return "out_of_scope"
        if r["has_question"] or r["intent"] == "product_question":
            return "retrieve"
        return "smalltalk"

    # ------------------------------------------------------------------ lead collection
    async def extract_lead(self, state: AgentState) -> dict:
        message = _last_human(state)
        ext = await self.brain.extract_lead(message, _last_ai(state))
        clean, rejected = ground_extraction(ext, message)
        lead = dict(state.get("lead") or {})
        changed = {k: v for k, v in clean.items() if lead.get(k) != v}
        lead.update(clean)
        update: dict[str, Any] = {"lead": lead, "lead_rejected": rejected, "turn_trace": ["extract_lead"]}
        if changed and state.get("lead_status") == "awaiting_confirmation":
            update["lead_status"] = "collecting"   # details changed -> must confirm again
        if state.get("lead_status") in (None, "none"):
            update["lead_status"] = "collecting"
        log.info("lead extraction: new=%s rejected=%s", list(changed), rejected)
        return update

    def after_extract(self, state: AgentState) -> str:
        return "retrieve" if state["route"].get("has_question") else "lead_next_step"

    def _lead_prompt(self, state: AgentState) -> tuple[str, dict]:
        """Deterministic next question for lead collection (+ state updates)."""
        lead, lang = state.get("lead") or {}, state.get("language", "th")
        missing = missing_fields(lead)
        rejected = [f for f in (state.get("lead_rejected") or []) if f in missing]
        labels = FIELD_LABEL_EN if lang == "en" else FIELD_LABEL_TH
        product = _product_name(lead.get("product_interest"))
        if missing:
            need = ", ".join(labels[f] for f in missing)
            if lang == "en":
                head = ("Great! So that an InsureX agent can contact you" + (f" about {product}" if product else "") +
                        ", could you share your " if len(missing) == 4 else "Thank you. Could you also share your ")
                text = head + need + "?"
                if rejected:
                    text = f"Sorry, I couldn't validate your {', '.join(labels[f] for f in rejected)}. " + text
            else:
                if len(missing) == 4:
                    text = ("ยินดีเลยครับ เพื่อให้ตัวแทนของ InsureX ติดต่อกลับไปให้ข้อมูล" +
                            (f"แผน {product} " if product else "") +
                            f"และนำเสนอแผนที่เหมาะกับคุณ รบกวนขอข้อมูลต่อไปนี้ครับ: {need}")
                else:
                    name = lead.get("full_name")
                    text = f"ขอบคุณครับ{' คุณ' + name if name else ''} รบกวนขอ{need}เพิ่มเติมด้วยครับ"
                if rejected:
                    text = f"ขออภัยครับ {', '.join(labels[f] for f in rejected)} ที่ให้มายังไม่ถูกต้องหรือไม่ชัดเจน " + text
            return text, {}
        summary = format_lead_summary(lead, lang, product)
        if lang == "en":
            text = (f"Please confirm your details:\n{summary}\n\nDo you confirm these details and consent to InsureX "
                    "contacting you about insurance products (PDPA)? Reply 'yes' to confirm, or tell me what to change.")
        else:
            text = (f"รบกวนตรวจสอบข้อมูลครับ\n{summary}\n\nยืนยันว่าข้อมูลถูกต้อง และยินยอมให้ InsureX "
                    "ติดต่อกลับเพื่อนำเสนอผลิตภัณฑ์ประกัน (ตาม พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล) ใช่ไหมครับ? "
                    "พิมพ์ 'ยืนยัน' หรือแจ้งส่วนที่ต้องการแก้ไขได้เลยครับ")
        return text, {"lead_status": "awaiting_confirmation"}

    async def lead_next_step(self, state: AgentState) -> dict:
        text, upd = self._lead_prompt(state)
        return {"answer": text, **upd, "turn_trace": ["lead_next_step"]}

    async def save_lead(self, state: AgentState) -> dict:
        lead, lang = state.get("lead") or {}, state.get("language", "th")
        sid = state.get("session_id") or "unknown"
        try:
            record = LeadRecord(session_id=sid, consent=True, **{k: v for k, v in lead.items()
                                                                  if k in LeadRecord.model_fields})
        except ValidationError as e:
            bad = {str(err["loc"][0]) for err in e.errors() if err["loc"]}
            for f in bad:
                lead.pop(f, None)
            text, upd = self._lead_prompt({**state, "lead": lead, "lead_rejected": list(bad)})
            return {"lead": lead, "lead_status": "collecting", "answer": text, **upd, "turn_trace": ["save_lead"]}
        try:
            result = await self.gateway.save_lead(record)          # -> MCP tool `save_lead`
        except Exception as e:                                       # transport / server crash
            log.exception("save_lead tool failed")
            msg = ("Sorry, our system could not save your details just now. Please reply 'yes' again in a moment."
                   if lang == "en" else
                   "ขออภัยครับ ระบบบันทึกข้อมูลขัดข้องชั่วคราว ข้อมูลของคุณยังอยู่ครบ รบกวนพิมพ์ 'ยืนยัน' อีกครั้งในอีกสักครู่ครับ")
            return {"answer": msg, "turn_trace": ["save_lead", f"tool_error:{type(e).__name__}"]}
        if not result.get("ok"):
            errors = "; ".join(result.get("errors", []))
            log.warning("save_lead rejected by server: %s", errors)
            return {"lead_status": "collecting",
                    "answer": ("ขออภัยครับ ข้อมูลบางส่วนไม่ผ่านการตรวจสอบ: " if lang == "th" else
                               "Some details failed validation: ") + errors,
                    "turn_trace": ["save_lead", "tool_rejected"]}
        lead_id = result["lead_id"]
        product = _product_name(lead.get("product_interest"))
        if lang == "en":
            text = (f"Thank you, {record.full_name}! Your details are saved (reference {lead_id}). An InsureX agent will "
                    f"contact you at {record.phone}" + (f" about {product}" if product else "") +
                    ". Anything else I can help with?")
        else:
            text = (f"ขอบคุณครับคุณ{record.full_name} บันทึกข้อมูลเรียบร้อยแล้ว (เลขอ้างอิง {lead_id}) "
                    f"ตัวแทนของ InsureX จะติดต่อกลับที่เบอร์ {record.phone}" + (f" เรื่องแผน {product}" if product else "") +
                    " ครับ มีอะไรให้ช่วยเพิ่มเติมไหมครับ")
        return {"lead_status": "saved", "lead_id": lead_id, "mode": "qa", "answer": text,
                "turn_trace": ["save_lead", f"mcp:{result.get('status')}"]}

    async def cancel_lead(self, state: AgentState) -> dict:
        lang = state.get("language", "th")
        text = ("No problem - I won't keep your details. Feel free to ask me anything about our plans."
                if lang == "en" else "ได้เลยครับ ผมจะไม่บันทึกข้อมูลของคุณ หากมีคำถามเรื่องแผนประกันสอบถามได้ตลอดครับ")
        return {"lead": {}, "lead_status": "cancelled", "mode": "qa", "answer": text, "turn_trace": ["cancel_lead"]}

    # ------------------------------------------------------------------ RAG
    async def retrieve(self, state: AgentState) -> dict:
        attempt = state.get("retrieval_attempt", 0) + 1
        query = state.get("query") or _last_human(state)
        products = state["route"].get("products") or []   # resolved by the router (incl. history)
        # 1st attempt: hard product filter; retry: no filter (maybe the product guess was wrong)
        res = await asyncio.to_thread(self.retriever.retrieve, query, products, None, attempt == 1)
        return {"retrieval_attempt": attempt, "retrieved": [_serialize(p) for p in res.parents],
                "turn_trace": [f"retrieve#{attempt}"]}

    async def grade_documents(self, state: AgentState) -> dict:
        docs = state.get("retrieved") or []
        if not docs:
            return {"relevant": [], "turn_trace": ["grade_documents:empty"]}
        # search with the (possibly rewritten) query, but judge relevance against what the customer asked:
        # a retry rewrite into brochure vocabulary can drop the off-topic word ("รถยนต์")
        asked = (state.get("route") or {}).get("standalone_query") or _last_human(state)
        g = await self.brain.grade(asked, docs)
        listed = [docs[i - 1] for i in g.relevant_blocks]
        # Not fully answerable -> corrective retry. On the last attempt, partial evidence beats a
        # refusal (e.g. "ต่างกันยังไง"): the generator states what the brochures do not specify and
        # the grounding check still applies. "No answer" now means nothing relevant was found.
        last_attempt = state.get("retrieval_attempt", 0) >= self.s.max_retrieval_attempts
        relevant = listed if (g.answerable or last_attempt) else []
        tag = "partial " if relevant and not g.answerable else ""
        return {"relevant": relevant, "turn_trace": [f"grade_documents:{tag}{len(relevant)}/{len(docs)}"]}

    def after_grade(self, state: AgentState) -> str:
        if state.get("relevant"):
            return "generate"
        if state.get("retrieval_attempt", 0) < self.s.max_retrieval_attempts:
            return "rewrite_query"
        return "no_answer"

    async def rewrite_query(self, state: AgentState) -> dict:
        new_q = await self.brain.retry_query(state.get("query") or _last_human(state))
        return {"query": new_q, "turn_trace": ["rewrite_query"]}

    async def generate(self, state: AgentState) -> dict:
        v = state.get("verification") or {}
        feedback = ""
        if v and not v.get("grounded", True):
            feedback = "Unsupported: " + "; ".join(v.get("unsupported_numbers", []) + v.get("unsupported_claims", []))
        question = _last_human(state)
        if state.get("query") and state["query"] != question:
            question = f"{question}\n(standalone: {state['query']})"
        a = await self.brain.generate(question, state["relevant"], state.get("messages", [])[:-1],
                                      state.get("language", "th"), feedback)
        return {"draft": a.model_dump(), "generation_attempt": state.get("generation_attempt", 0) + 1,
                "turn_trace": [f"generate#{state.get('generation_attempt', 0) + 1}"]}

    async def verify_answer(self, state: AgentState) -> dict:
        draft, relevant = state["draft"], state["relevant"]
        ctx = format_context(relevant)
        bad_numbers = unsupported_numbers(draft["answer"], ctx, _last_human(state))
        claims: list[str] = []
        grounded = not bad_numbers
        if grounded and self.s.verify_with_llm and isinstance(self.brain, LLMBrain):
            g = await self.brain.check_grounding(draft["answer"], ctx)
            grounded, claims = g.grounded, g.unsupported_claims
        verdict = {"grounded": grounded, "unsupported_numbers": bad_numbers, "unsupported_claims": claims}
        return {"verification": verdict, "turn_trace": [f"verify_answer:{'ok' if grounded else 'fail'}"]}

    def after_verify(self, state: AgentState) -> str:
        if state["verification"]["grounded"]:
            return "finalize_answer"
        if state.get("generation_attempt", 0) < self.s.max_generation_attempts:
            return "generate"
        return "no_answer"

    async def finalize_answer(self, state: AgentState) -> dict:
        draft, relevant, lang = state["draft"], state["relevant"], state.get("language", "th")
        cited = draft.get("cited_blocks") or list(range(1, len(relevant) + 1))
        sources = [f"[{i}] {relevant[i - 1]['citation']}" for i in cited if 1 <= i <= len(relevant)]
        text = draft["answer"].strip()
        if not draft.get("fully_answered", True):
            text += ("\n\nFor details not covered in the brochures, an InsureX agent can help - just say you're interested."
                     if lang == "en" else
                     f"\n\nสำหรับรายละเอียดที่เอกสารไม่ได้ระบุ แนะนำสอบถาม {self.s.contact_center} "
                     "หรือพิมพ์ 'สนใจ' เพื่อให้ตัวแทนติดต่อกลับได้ครับ")
        if sources:
            text += ("\n\nSources:\n" if lang == "en" else "\n\nแหล่งข้อมูล:\n") + "\n".join(sources)
        products: list[str] = []
        for i in cited:
            if 1 <= i <= len(relevant):
                pid = relevant[i - 1]["metadata"]["product_id"]
                if pid not in products:
                    products.append(pid)
        upd: dict[str, Any] = {"answer": text, "sources": sources, "turn_trace": ["finalize_answer"]}
        if products:
            upd["active_products"] = products[:2]
        return upd

    async def no_answer(self, state: AgentState) -> dict:
        """Error handling for 'answer not found' (or answer could not be verified)."""
        lang = state.get("language", "th")
        names = ", ".join(p.name_th for p in PRODUCTS)
        if lang == "en":
            text = ("Sorry, I couldn't find that information in the product brochures I have "
                    f"({names}), so I'd rather not guess. Please contact {self.s.contact_center}, "
                    "or say you're interested and an agent will call you back.")
        else:
            text = ("ขออภัยครับ ผมไม่พบข้อมูลเรื่องนี้ในเอกสารผลิตภัณฑ์ที่มีอยู่ "
                    f"({names}) จึงขอไม่ตอบแบบคาดเดานะครับ แนะนำสอบถาม {self.s.contact_center} "
                    "หรือพิมพ์ 'สนใจ' เพื่อให้ตัวแทนติดต่อกลับได้เลยครับ")
        reason = "unverified" if state.get("draft") else "not_found"
        return {"answer": text, "turn_trace": [f"no_answer:{reason}"]}

    async def smalltalk(self, state: AgentState) -> dict:
        lang, msg = state.get("language", "th"), _last_human(state)
        thanks = any(w in msg.lower() for w in ("ขอบคุณ", "ขอบใจ", "thank"))
        if lang == "en":
            text = ("You're welcome! " if thanks else "Hello! I'm the InsureX sales assistant. ") + \
                   "I can explain our life-protection plans (" + ", ".join(p.name_en for p in PRODUCTS) + \
                   "), compare them, and arrange for an agent to contact you."
        else:
            text = ("ยินดีครับ " if thanks else "สวัสดีครับ ผมเป็นผู้ช่วยขายของ InsureX ") + \
                   "ผมช่วยอธิบายและเปรียบเทียบแผนประกันชีวิต (" + ", ".join(p.name_th for p in PRODUCTS) + \
                   ") และประสานให้ตัวแทนติดต่อกลับได้ครับ สนใจสอบถามเรื่องไหนดีครับ"
        return {"answer": text, "turn_trace": ["smalltalk"]}

    async def catalog(self, state: AgentState) -> dict:
        """'What products do you have?' is master data, not a brochure search: answer from the catalogue."""
        if state.get("language", "th") == "en":
            text = (f"InsureX offers {len(PRODUCTS)} life-insurance plans:\n{catalog_overview_en()}\n\n"
                    "Ask me about any of them (coverage, conditions, example premiums), or say you're interested "
                    "and an agent will contact you.")
        else:
            text = (f"InsureX มีแผนประกันชีวิต {len(PRODUCTS)} แบบครับ\n{catalog_overview_th()}\n\n"
                    "สนใจแผนไหน ถามรายละเอียดเพิ่มได้เลยครับ เช่น ความคุ้มครอง เงื่อนไข หรือตัวอย่างเบี้ย "
                    "หรือพิมพ์ 'สนใจ' เพื่อให้ตัวแทนติดต่อกลับ")
        return {"answer": text, "turn_trace": ["catalog"]}

    async def unknown_plan(self, state: AgentState) -> dict:
        codes = ", ".join(state["route"]["unknown_plan_codes"])
        if state.get("language", "th") == "en":
            text = (f"InsureX has no plan {codes}. Our plans are:\n{catalog_overview_en()}\n\n"
                    "Which one did you mean?")
        else:
            text = (f"ไม่พบแบบประกัน {codes} ใน InsureX ครับ แผนที่มีคือ\n{catalog_overview_th()}\n\n"
                    "หมายถึงแผนไหนครับ")
        return {"answer": text, "turn_trace": [f"unknown_plan:{codes}"]}

    async def out_of_scope(self, state: AgentState) -> dict:
        lang = state.get("language", "th")
        text = ("I can only help with InsureX life-insurance products. Is there anything about our plans I can help with?"
                if lang == "en" else
                "ขออภัยครับ ผมช่วยได้เฉพาะเรื่องผลิตภัณฑ์ประกันชีวิตของ InsureX มีเรื่องแผนประกันที่อยากสอบถามไหมครับ")
        return {"answer": text, "turn_trace": ["out_of_scope"]}

    # ------------------------------------------------------------------ finalize
    async def finalize(self, state: AgentState) -> dict:
        text = state.get("answer") or ""
        upd: dict[str, Any] = {}
        trace = state.get("turn_trace") or []
        lead_handled = any(t.startswith(("lead_next_step", "save_lead", "cancel_lead")) for t in trace)
        if state.get("mode") == "lead_collection" and not lead_handled and \
                state.get("lead_status") in ("collecting", "awaiting_confirmation"):
            follow, upd = self._lead_prompt(state)   # keep collecting after a Q&A detour
            text = f"{text}\n\n{follow}" if text else follow
        upd.update({"messages": [AIMessage(text)], "answer": text, "turn_trace": ["finalize"]})
        return upd

    # ------------------------------------------------------------------ build
    def build(self, checkpointer=None):
        g = StateGraph(AgentState)
        for name in ("prepare_turn", "route", "extract_lead", "lead_next_step", "save_lead", "cancel_lead",
                     "retrieve", "grade_documents", "rewrite_query", "generate", "verify_answer",
                     "finalize_answer", "no_answer", "smalltalk", "catalog", "unknown_plan", "out_of_scope",
                     "finalize"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "prepare_turn")
        g.add_edge("prepare_turn", "route")
        g.add_conditional_edges("route", self.after_route,
                                ["save_lead", "cancel_lead", "unknown_plan", "catalog", "extract_lead", "retrieve",
                                 "smalltalk", "out_of_scope"])
        g.add_conditional_edges("extract_lead", self.after_extract, ["retrieve", "lead_next_step"])
        g.add_edge("retrieve", "grade_documents")
        g.add_conditional_edges("grade_documents", self.after_grade, ["generate", "rewrite_query", "no_answer"])
        g.add_edge("rewrite_query", "retrieve")
        g.add_edge("generate", "verify_answer")
        g.add_conditional_edges("verify_answer", self.after_verify, ["finalize_answer", "generate", "no_answer"])
        for n in ("lead_next_step", "save_lead", "cancel_lead", "finalize_answer", "no_answer", "smalltalk",
                  "catalog", "unknown_plan", "out_of_scope"):
            g.add_edge(n, "finalize")
        g.add_edge("finalize", END)
        return g.compile(checkpointer=checkpointer)
