"""The 'brain': every step that needs language understanding.

LLMBrain   - real LLM with structured (Pydantic) outputs.
RuleBrain  - deterministic heuristics. Used (a) when LLM_PROVIDER=offline (tests,
             CI, demo without API keys) and (b) as a per-call FALLBACK when the
             LLM call fails (timeout, rate limit, malformed output). The agent
             degrades instead of crashing, and the turn is flagged `degraded`.

Graph nodes only talk to this interface, so the workflow logic is identical in
both modes and can be unit-tested without network access.
"""

from __future__ import annotations

import contextvars
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from ..config import Settings, get_settings
from ..ingestion.catalog import OTHER_INSURANCE_RE, PRODUCT_BY_ID, SYNONYMS, detect_products, expand_query
from ..ingestion.thai_text import parse_thai_amount, tokenize
from ..leads.models import LeadExtraction, find_phone_in_text
from . import prompts
from .schemas import GeneratedAnswer, GradeResult, GroundingCheck, RetryQuery, RouteDecision

log = logging.getLogger(__name__)

# steps that fell back to rules during the current turn (set per turn by the app)
DEGRADED_STEPS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar("degraded_steps", default=None)


# ----------------------------------------------------------------------------- helpers
def detect_language(text: str) -> str:
    thai = sum(1 for c in text if "฀" <= c <= "๿")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return "th" if thai >= latin * 0.3 or thai > 3 else "en"


def format_context(parents: list[dict], max_chars: int = 2800) -> str:
    """Numbered context blocks. Each block lists the best-matching rows first (the
    child chunks that retrieval hit) so they survive truncation of long tables."""
    blocks = []
    for i, p in enumerate(parents, 1):
        m = p["metadata"]
        text = p["text"] if len(p["text"]) <= max_chars else p["text"][:max_chars] + " …"
        hits = [h["text"] for h in p.get("hits", []) if h.get("chunk_type") != "section"][:3]
        key = ("Best-matching lines:\n" + "\n".join(f"- {h}" for h in hits) + "\nFull section:\n") if hits else ""
        blocks.append(f"[{i}] {m['product_name']} | {m['section_title']} | {m['source']} หน้า {m['page']}\n{key}{text}")
    return "\n\n".join(blocks)


def format_history(messages: list[AnyMessage], limit: int) -> str:
    lines = []
    for m in messages[-limit:]:
        role = "Customer" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) or "(no earlier messages)"


_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ALWAYS_ALLOWED = {"1314", "1351", "100"}   # contact-centre numbers, "100%"


def unsupported_numbers(answer: str, context: str, question: str = "") -> list[str]:
    """Deterministic hallucination guard: every amount >= 1,000 and every percentage
    in the answer must literally appear in the context (or the question)."""
    pool = {n.replace(",", "") for n in _NUM_RE.findall(context + " " + question)} | _ALWAYS_ALLOWED
    bad = []
    for raw in _NUM_RE.findall(re.sub(r"\[\d+\]", " ", answer)):
        n = raw.replace(",", "").rstrip(".")
        try:
            val = float(n)
        except ValueError:
            continue
        is_pct = bool(re.search(re.escape(raw) + r"\s*%", answer))
        if (val >= 1000 or is_pct) and n not in pool:
            bad.append(raw)
    return sorted(set(bad))


# ----------------------------------------------------------------------------- rule brain
_CONFIRM_RE = re.compile(r"^\s*(ใช่|ถูกต้อง|ยืนยัน|ตกลง|ยินยอม|ได้เลย|ได้ครับ|ได้ค่ะ|โอเค|ok|okay|yes|yep|confirm|correct)(?:\s*(?:ครับ|ค่ะ|คะ|จ้า|นะ|เลย|ทุกอย่าง|ข้อมูลถูกต้อง))*\s*[.!]*\s*$", re.I)
_CANCEL_RE = re.compile(r"ยกเลิก|ไม่สนใจแล้ว|ไม่ต้องติดต่อ|ไม่สะดวกให้ข้อมูล|ไม่ให้ข้อมูล|ไม่ยินยอม|cancel|stop|never ?mind|not interested", re.I)
_INTEREST_RE = re.compile(r"(?<!น่า)สนใจ(?!ไหม|มั้ย|หรือ)|อยากสมัคร|อยากซื้อ|อยากทำประกัน|สมัครเลย|ติดต่อกลับ|ให้เจ้าหน้าที่|ขอคุยกับตัวแทน|ให้ตัวแทน|interested|sign me up|call me|contact me|want to buy|apply", re.I)
_QUESTION_RE = re.compile(r"\?|ไหม|มั้ย|อะไร|เท่าไ[รห]|อย่างไร|ยังไง|กี่|บ้าง|หรือเปล่า|หรือไม่|แบบไหน|ตัวไหน|อันไหน|ได้ไหม|คืออะไร|ต่างกัน|เปรียบเทียบ|ขอข้อมูล|อยากรู้|อยากทราบ|\b(how|what|which|can|does|is|are|do|tell me|compare)\b", re.I)
_SMALLTALK_RE = re.compile(r"^\s*(สวัสดี|หวัดดี|ขอบคุณ|ขอบใจ|thank|thanks|hello|hi|hey)(?=$|\s|[!.,?]|ครับ|ค่ะ|คะ|จ้า|นะ|มาก)", re.I)
_CATALOG_RE = re.compile(r"มี(?:ประกัน|แผน|แบบ|ผลิตภัณฑ์)\S{0,8}อะไรบ้าง|ขาย(?:ประกัน|แผน)?\S{0,8}อะไรบ้าง|"
                         r"\bwhat (?:products|plans|insurance)\b|\bwhich (?:products|plans)\b", re.I)
_CORRECT_RE = re.compile(r"แก้|เปลี่ยน|ไม่ใช่|ผิด|wrong|change|update", re.I)
_PERSONAL_RE = re.compile(r"ชื่อ|อาชีพ|ทำงาน|รายได้|เงินเดือน|เบอร์|โทร|name|occupation|income|salary|phone", re.I)

_FOLLOWUP_RE = re.compile(r"^\s*(แล้ว|ส่วน|และ)|นี้|นั้น|ดังกล่าว|ตัวเดิม|แผนเดิม|\b(it|this|that|same)\b", re.I)
_QWORDS = {"เท่าไหร่", "เท่าไร", "ไหม", "มั้ย", "ยังไง", "อย่างไร", "บ้าง", "หรือเปล่า", "หรือไม่", "ได้", "แล้ว",
           "ครับ", "ค่ะ", "คะ", "how", "much", "many", "what", "which", "does", "do", "can", "is"}
try:  # Thai function words never carry topical meaning
    from pythainlp.corpus import thai_stopwords as _thai_stopwords
    _QWORDS |= set(_thai_stopwords()) | {"เท่าไร", "เท่าไหร่", "กี่", "ได้ไหม", "หรือเปล่า", "ตั้งแต่", "ถึง"}
except Exception:  # pragma: no cover
    pass
_GENERIC_VERBS = {"ได้เงิน", "สมัคร", "ซื้อ", "อยากได้", "แนะนำ", "เลือก", "ทำประกัน", "ทำได้", "ได้รับ", "คุ้มไหม"}
_STOP_FIELD = r"(?=\s|,|$|ครับ|ค่ะ|คะ|นะ)"
_PERSONAL_SPAN_RE = re.compile(
    r"(?:ผม|ดิฉัน|ฉัน|หนู)?ชื่อ(?:-นามสกุล)?\s*[:：]?\s*\S+(?:\s+[^\s0-9]+)?|อาชีพ\s*\S+|ทำงานเป็น\s*\S+|"
    r"(?:รายได้|เงินเดือน)\D{0,10}[\d,\.]+\s*(?:ล้าน|แสน|หมื่น|พัน|k|K)?(?:ต่อเดือน|ต่อปี)?|"
    r"(?:เบอร์(?:โทร)?|โทร)\s*[:：]?\s*[\d\s\-+]{9,}|(?:\+?66|0)[\d\s\-]{8,12}")


class RuleBrain:
    name = "rules"

    async def route(self, message: str, history: list[AnyMessage], state: dict) -> RouteDecision:
        lang = detect_language(message)
        products = detect_products(message)
        awaiting = state.get("lead_status") == "awaiting_confirmation"
        has_q = bool(_QUESTION_RE.search(message))
        wants = bool(_INTEREST_RE.search(message))
        if awaiting and _CONFIRM_RE.search(message):
            intent = "confirm"
        elif _CANCEL_RE.search(message):
            intent = "cancel"
        elif wants:
            intent = "purchase_interest"
        elif find_phone_in_text(message) or (state.get("mode") == "lead_collection" and _PERSONAL_RE.search(message)):
            intent = "correct" if awaiting and _CORRECT_RE.search(message) else "provide_info"
        elif state.get("mode") == "lead_collection" and not has_q:
            intent = "provide_info"
        elif _SMALLTALK_RE.search(message) and not has_q:
            intent = "smalltalk"
        elif _CATALOG_RE.search(message) and not products and not OTHER_INSURANCE_RE.search(message):
            intent, has_q = "catalog", False
        else:
            intent = "product_question"
            has_q = True
        if not products and has_q and _FOLLOWUP_RE.search(message):
            products = list(state.get("active_products") or [])[:2]   # "แล้วแผนนี้ ..." -> product in focus
        query = ""
        if has_q:
            message = _PERSONAL_SPAN_RE.sub(" ", message).strip()   # personal data never goes into search
            names = " ".join(PRODUCT_BY_ID[p].name_th for p in products if p not in detect_products(message))
            query = f"{message} {names}".strip()
        return RouteDecision(intent=intent, wants_contact=wants, has_question=has_q and intent != "confirm",
                             products=products, standalone_query=query, language=lang)

    async def retry_query(self, query: str) -> str:
        swaps = {"ทุนประกัน": "จำนวนเงินเอาประกันภัย", "ค่าเบี้ย": "เบี้ยประกันภัย", "ตาย": "เสียชีวิต",
                 "จ่ายกี่ปี": "ระยะเวลาชำระเบี้ยประกันภัย", "รับอายุ": "อายุรับประกันภัย"}
        for a, b in swaps.items():
            query = query.replace(a, b)
        return query

    def bind_corpus(self, texts: list[str]) -> None:
        """Document frequencies of the KB, so the rule grader can weight rare words."""
        from collections import Counter
        self._n_docs = len(texts)
        self._df = Counter(t for txt in texts for t in set(tokenize(txt)))

    def _weight(self, tok: str) -> float:
        df, n = getattr(self, "_df", None), getattr(self, "_n_docs", 0)
        if not df:
            return 1.0
        if tok not in df:
            return 1.5          # word the KB has never seen (e.g. "รถยนต์") - strong signal of no answer
        return max(0.1, 1.0 - df[tok] / n)

    async def grade(self, question: str, parents: list[dict]) -> GradeResult:
        product_toks = {t for p in PRODUCT_BY_ID.values() for t in tokenize(p.name_th + " " + " ".join(p.aliases))}
        q = [t for t in set(tokenize(question)) if t not in product_toks and t not in _QWORDS]
        q_exp = [t for t in set(tokenize(expand_query(question))) if t not in product_toks and t not in _QWORDS]
        asked = detect_products(question)
        df = getattr(self, "_df", None) or {}
        # content words the KB has never seen and that are not known colloquialisms
        # ("รถยนต์") must be found, otherwise the KB cannot answer this question
        # (only when no product of ours is named - otherwise the question is about the KB by definition)
        unknown = [t for t in q if df and not asked and t not in df and len(t) >= 3 and not t.isdigit()
                   and t not in _GENERIC_VERBS and not any(t in k or k in t for k in SYNONYMS)]
        covs = []
        for i, p in enumerate(parents, 1):
            if asked and p["metadata"]["product_id"] not in asked and len(asked) == 1:
                covs.append(0.0)              # block about another product is not relevant
                continue
            body = (p["text"] + " " + " ".join(h["text"] for h in p.get("hits", []))).lower().replace(",", "")
            # substring containment is robust to Thai segmentation differences
            def cov(tokens):
                total = sum(self._weight(t) for t in tokens) or 1.0
                return sum(self._weight(t) for t in tokens if t in body) / total
            if unknown and not any(t in body for t in unknown):
                covs.append(0.0)
                continue
            covs.append(max(cov(q), cov(q_exp)))
        best = max(covs, default=0.0)
        # the rule grader is only a sanity gate ("is anything here on-topic?");
        # ranking is left to the hybrid retriever, which is far better at it
        relevant = [i for i, c in enumerate(covs, 1) if c >= max(0.3, 0.6 * best)]
        return GradeResult(relevant_blocks=relevant[:2], answerable=bool(relevant))

    async def generate(self, question: str, parents: list[dict], history: list[AnyMessage], language: str,
                 feedback: str = "") -> GeneratedAnswer:
        q = set(tokenize(expand_query(question))) - _QWORDS
        # extractive: walk blocks in the grader's order and hits in the retriever's (RRF) order
        picked, seen = [], set()
        for i, p in enumerate(parents, 1):
            hits = [h for h in p.get("hits", []) if h.get("chunk_type") not in ("section", "legal")] \
                or p.get("hits", [])
            for h in hits:
                body = h["text"].split(" > ")[-1].lstrip("- ").strip()
                body = body if len(body) <= 260 else body[:260] + "…"
                if body in seen or not any(t in h["text"].lower().replace(",", "") for t in q):
                    continue
                picked.append((i, body))
                seen.add(body)
                if len(picked) == 3:
                    break
            if len(picked) == 3:
                break
        intro = ("ข้อมูลจากเอกสารผลิตภัณฑ์ที่เกี่ยวข้องครับ (โหมดออฟไลน์: แสดงข้อความจากเอกสารโดยตรง)"
                 if language == "th" else "Relevant passages from the product brochures (offline mode, extractive):")
        lines = [intro] + [f"- {body} [{i}]" for i, body in picked]
        return GeneratedAnswer(answer="\n".join(lines), cited_blocks=sorted({i for i, _ in picked}),
                               fully_answered=bool(picked))

    async def check_grounding(self, answer: str, context: str) -> GroundingCheck:
        return GroundingCheck(grounded=True, unsupported_claims=[])

    async def extract_lead(self, message: str, last_assistant: str = "") -> LeadExtraction:
        out: dict[str, Any] = {}
        m = re.search(r"(?:ชื่อ(?:-นามสกุล)?|ผมชื่อ|ดิฉันชื่อ|ฉันชื่อ|my name is|name is|name:)\s*[:：]?\s*(?:คุณ|นาย|นางสาว|นาง)?\s*"
                      r"([^\s,0-9]+(?:\s+[^\s,0-9]+)?)" + _STOP_FIELD, message, re.I)
        if m:
            name = re.sub(r"(ครับ|ค่ะ|คะ|นะ)$", "", m.group(1).strip())
            out["full_name"] = re.split(r"\s+(?:อาชีพ|ทำงาน|รายได้|เงินเดือน|เบอร์)", name)[0]
        m = re.search(r"(?:อาชีพ|ทำงานเป็น|ทำอาชีพ|ทำงาน|occupation|work as an?|i am an?|i'm an?)\s*[:：]?\s*([^\s,0-9]+(?:\s[^\s,0-9]+)?)" + _STOP_FIELD,
                      message, re.I)
        if m:
            out["occupation"] = re.sub(r"(ครับ|ค่ะ|คะ|นะ)$", "", m.group(1).split(" ")[0])
        m = re.search(r"(?:รายได้|เงินเดือน|income|salary)[^\d๐-๙]{0,20}([\d๐-๙][\d,\.๐-๙]*\s*(?:ล้าน|แสน|หมื่น|พัน|k|K)?(?:\s*\d+)?)",
                      message, re.I)
        if m and parse_thai_amount(m.group(1)):
            out["income_text"] = m.group(1).strip()
            out["income_amount"] = parse_thai_amount(m.group(1))
            if re.search(r"ต่อปี|ปีละ|/ปี|per year|annual", message, re.I):
                out["income_period"] = "yearly"
            elif re.search(r"ต่อเดือน|เดือนละ|/เดือน|เงินเดือน|per month|monthly", message, re.I):
                out["income_period"] = "monthly"
        phone = find_phone_in_text(message)
        if phone:
            out["phone"] = phone
        # bare answers to the assistant's last question ("วิศวกรครับ", "30000")
        if not out and last_assistant and "รบกวนขอ" in last_assistant:
            bare = re.sub(r"(ครับ|ค่ะ|คะ|นะ|จ้า)$", "", message.strip()).strip()
            if "อาชีพ" in last_assistant and len(bare) <= 30 and not re.search(r"\d", bare):
                out["occupation"] = bare
            elif "ชื่อ" in last_assistant and len(bare) <= 40 and not re.search(r"\d", bare):
                out["full_name"] = bare
            elif "รายได้" in last_assistant and parse_thai_amount(bare):
                out["income_text"], out["income_amount"] = bare, parse_thai_amount(bare)
        return LeadExtraction(**out)

    async def summarize(self, previous_summary: str, messages: list[AnyMessage]) -> str:
        text = " | ".join(str(m.content)[:120] for m in messages if isinstance(m, HumanMessage))
        return (previous_summary + " | " + text)[-1200:].strip(" |")


# ----------------------------------------------------------------------------- LLM brain
class LLMBrain:
    """LLM-backed implementation; any failure falls back to RuleBrain for that call."""

    name = "llm"

    def __init__(self, model, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.model = model
        self.fallback = RuleBrain()
        self.degraded_calls: list[str] = []

    async def _structured(self, schema, messages):
        return await self.model.with_structured_output(schema).ainvoke(messages)

    async def _safe(self, step: str, call, fallback):
        try:
            out = await call()
            if out is None:
                raise ValueError("model returned no structured output")
            return out
        except Exception as e:  # network, rate limit, validation ...
            log.warning("LLM step '%s' failed (%s: %s) -> rule-based fallback", step, type(e).__name__, e)
            self.degraded_calls.append(step)
            steps = DEGRADED_STEPS.get()
            if steps is not None:
                steps.append(step)
            return await fallback()

    # -- steps ------------------------------------------------------------------------
    async def route(self, message: str, history: list[AnyMessage], state: dict) -> RouteDecision:
        from ..leads.models import missing_fields

        sys = prompts.ROUTER_SYSTEM.format(
            mode=state.get("mode", "qa"), lead_status=state.get("lead_status", "none"),
            missing=", ".join(missing_fields(state.get("lead") or {})) or "-",
            active_products=", ".join(state.get("active_products") or []) or "-",
            summary=state.get("summary") or "-",
        )
        msgs = [SystemMessage(sys),
                HumanMessage(f"Conversation so far:\n{format_history(history, self.s.history_window)}\n\n"
                             f"LATEST customer message:\n{message}")]

        async def call():
            d: RouteDecision = await self._structured(RouteDecision, msgs)
            d.products = [p for p in d.products if p in PRODUCT_BY_ID]
            if d.has_question and not d.standalone_query:
                d.standalone_query = message
            return d

        return await self._safe("route", call, lambda: self.fallback.route(message, history, state))

    async def retry_query(self, query: str) -> str:
        msgs = [SystemMessage(prompts.RETRY_SYSTEM), HumanMessage(query)]

        async def call():
            return (await self._structured(RetryQuery, msgs)).query

        return await self._safe("retry_query", call, lambda: self.fallback.retry_query(query))

    async def grade(self, question: str, parents: list[dict]) -> GradeResult:
        msgs = [SystemMessage(prompts.GRADER_SYSTEM),
                HumanMessage(f"QUESTION: {question}\n\nCONTEXT:\n{format_context(parents, 1500)}")]

        async def call():
            g: GradeResult = await self._structured(GradeResult, msgs)
            g.relevant_blocks = [i for i in g.relevant_blocks if 1 <= i <= len(parents)]
            return g

        return await self._safe("grade", call, lambda: self.fallback.grade(question, parents))

    async def generate(self, question: str, parents: list[dict], history: list[AnyMessage], language: str,
                       feedback: str = "") -> GeneratedAnswer:
        fb = f"\nA previous draft was rejected by the fact-checker: {feedback}\nFix it." if feedback else ""
        sys = prompts.GENERATOR_SYSTEM.format(language_name="Thai" if language == "th" else "English", feedback=fb)
        msgs = [SystemMessage(sys),
                HumanMessage(f"Recent conversation:\n{format_history(history, 6)}\n\n"
                             f"CONTEXT:\n{format_context(parents)}\n\nQUESTION: {question}")]

        async def call():
            a: GeneratedAnswer = await self._structured(GeneratedAnswer, msgs)
            a.cited_blocks = [i for i in a.cited_blocks if 1 <= i <= len(parents)]
            return a

        return await self._safe("generate", call,
                                lambda: self.fallback.generate(question, parents, history, language, feedback))

    async def check_grounding(self, answer: str, context: str) -> GroundingCheck:
        msgs = [SystemMessage(prompts.GROUNDING_SYSTEM), HumanMessage(f"CONTEXT:\n{context}\n\nANSWER:\n{answer}")]

        async def call():
            return await self._structured(GroundingCheck, msgs)

        return await self._safe("check_grounding", call, lambda: self.fallback.check_grounding(answer, context))

    async def extract_lead(self, message: str, last_assistant: str = "") -> LeadExtraction:
        msgs = [SystemMessage(prompts.EXTRACT_SYSTEM),
                HumanMessage(f"Assistant's previous message: {last_assistant or '-'}\n\nLATEST customer message: {message}")]

        async def call():
            return await self._structured(LeadExtraction, msgs)

        return await self._safe("extract_lead", call, lambda: self.fallback.extract_lead(message, last_assistant))

    async def summarize(self, previous_summary: str, messages: list[AnyMessage]) -> str:
        msgs = [SystemMessage(prompts.SUMMARY_SYSTEM),
                HumanMessage(f"Existing summary: {previous_summary or '-'}\n\nMessages:\n{format_history(messages, 100)}")]

        async def call():
            out = await self.model.ainvoke(msgs)
            content = out.content if isinstance(out, AIMessage) else str(out)
            if isinstance(content, list):  # some providers return content blocks
                content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
            return content

        return await self._safe("summarize", call, lambda: self.fallback.summarize(previous_summary, messages))


def make_brain(settings: Settings | None = None):
    from ..llm import get_chat_model

    s = settings or get_settings()
    model = get_chat_model(s)
    return RuleBrain() if model is None else LLMBrain(model, s)
