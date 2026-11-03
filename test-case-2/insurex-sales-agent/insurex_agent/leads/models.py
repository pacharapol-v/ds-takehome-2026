"""Lead data contracts.

`LeadExtraction` is what the LLM is allowed to produce (everything optional).
`LeadRecord` is what may be written to the database (everything validated).
The gap between the two is where data integrity is enforced:
  * phone numbers normalised to Thai national format and length-checked,
  * income parsed to a number and normalised to *monthly* baht,
  * each extracted value must literally appear in what the customer typed
    (see `ground_extraction`) - the model cannot invent a phone number,
  * PDPA consent must be explicit before a record is persisted.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ..ingestion.thai_text import parse_thai_amount

REQUIRED_FIELDS = ("full_name", "occupation", "monthly_income", "phone")
FIELD_LABEL_TH = {
    "full_name": "ชื่อ-นามสกุล",
    "occupation": "อาชีพ",
    "monthly_income": "รายได้ต่อเดือน",
    "phone": "เบอร์โทรศัพท์ติดต่อ",
}
FIELD_LABEL_EN = {
    "full_name": "full name",
    "occupation": "occupation",
    "monthly_income": "monthly income",
    "phone": "contact phone number",
}

def _no_default(schema: dict) -> None:  # OpenAI strict schemas do not allow `default`
    schema.pop("default", None)


_TITLE_RE = re.compile(r"^(คุณ|นาย|นางสาว|นาง|น\.ส\.|ด\.ช\.|ด\.ญ\.|mr\.?|mrs\.?|ms\.?|miss)\s*", re.I)


def normalize_phone(raw: str | None) -> str | None:
    """'081-234-5678', '+66 81 234 5678', '๐๘๑๒๓๔๕๖๗๘' -> '0812345678'. None if invalid."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")))
    if digits.startswith("66") and len(digits) in (10, 11):
        digits = "0" + digits[2:]
    if re.fullmatch(r"0[689]\d{8}", digits):      # mobile
        return digits
    if re.fullmatch(r"0[2-7]\d{7}", digits):       # landline (02 xxx xxxx, 053 xxx xxx ...)
        return digits
    return None


def find_phone_in_text(text: str) -> str | None:
    for m in re.finditer(r"(?:\+?66[\s-]?|0)[\d\s\-๐-๙]{8,13}", text or ""):
        p = normalize_phone(m.group(0))
        if p:
            return p
    return None


class LeadExtraction(BaseModel):
    """Fields the model may extract from the latest customer message. Use null when not stated."""

    full_name: Optional[str] = Field(None, description="Customer's name exactly as they wrote it, without titles like คุณ/นาย. null if not stated.")
    occupation: Optional[str] = Field(None, description="Customer's occupation/job as written (e.g. 'วิศวกร', 'ค้าขาย'). null if not stated.")
    income_text: Optional[str] = Field(None, description="The income phrase exactly as the customer wrote it, e.g. '3 หมื่น', '50,000 บาท/เดือน'. null if not stated.")
    income_amount: Optional[float] = Field(None, description="Income as a number in baht, e.g. '3 หมื่น' -> 30000. null if not stated.")
    income_period: Literal["monthly", "yearly", "unknown"] = Field("unknown", json_schema_extra=_no_default, description="Whether the income is per month or per year ('ต่อปี', 'ปีละ' => yearly).")
    phone: Optional[str] = Field(None, description="Phone number exactly as written. null if not stated.")


class LeadRecord(BaseModel):
    """A validated, consented lead ready to be persisted."""

    session_id: str = Field(min_length=1)
    full_name: str = Field(min_length=2, max_length=120)
    occupation: str = Field(min_length=2, max_length=120)
    monthly_income: float = Field(gt=0, lt=50_000_000)
    income_text: Optional[str] = None
    phone: str
    product_interest: Optional[str] = None
    consent: bool
    source: str = "sales-agent-chat"
    captured_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @field_validator("full_name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = _TITLE_RE.sub("", v.strip())
        if re.search(r"\d", v):
            raise ValueError("name must not contain digits")
        return re.sub(r"\s+", " ", v)

    @field_validator("occupation")
    @classmethod
    def _occ(cls, v: str) -> str:
        return re.sub(r"\s+", " ", v.strip())

    @field_validator("phone")
    @classmethod
    def _phone(cls, v: str) -> str:
        p = normalize_phone(v)
        if not p:
            raise ValueError(f"invalid Thai phone number: {v!r}")
        return p

    @model_validator(mode="after")
    def _consent(self) -> "LeadRecord":
        if not self.consent:
            raise ValueError("PDPA: explicit consent is required before storing a lead")
        return self


def _norm(s: str) -> str:
    return re.sub(r"[\s\-\.,]", "", (s or "")).lower()


def ground_extraction(ext: LeadExtraction, user_text: str) -> tuple[dict, list[str]]:
    """Keep only values that are supported by the customer's own words.

    Returns (clean_fields, rejected_field_names). `clean_fields` uses the
    LeadRecord field names (monthly_income already normalised to per-month).
    """
    clean: dict = {}
    rejected: list[str] = []
    text_n = _norm(user_text)

    for f in ("full_name", "occupation"):
        val = getattr(ext, f)
        if not val:
            continue
        core = _TITLE_RE.sub("", val.strip())
        if _norm(core) and _norm(core) in text_n:
            clean[f] = core.strip()
        else:
            rejected.append(f)

    if ext.phone:
        p = normalize_phone(ext.phone)
        in_text = find_phone_in_text(user_text)
        if p and in_text == p:
            clean["phone"] = p
        else:
            rejected.append("phone")
    elif find_phone_in_text(user_text):            # model missed an obvious number
        clean["phone"] = find_phone_in_text(user_text)

    if ext.income_amount or ext.income_text:
        amount = parse_thai_amount(ext.income_text) if ext.income_text else None
        if amount is None and ext.income_amount:
            amount = float(ext.income_amount)
        # the number (or its Thai phrase) must be visible in the message
        supported = bool(ext.income_text and _norm(ext.income_text) in text_n)
        if not supported and amount:
            supported = any(abs((parse_thai_amount(m.group(0)) or -1) - amount) < 1
                            for m in re.finditer(r"[\d,\.]+\s*(?:ล้าน|แสน|หมื่น|พัน|k|K)?", user_text))
        if amount and supported:
            period = ext.income_period
            if period == "unknown" and re.search(r"ต่อปี|ปีละ|/ปี|per year|annual|yearly", user_text, re.I):
                period = "yearly"
            clean["monthly_income"] = round(amount / 12, 2) if period == "yearly" else amount
            clean["income_text"] = ext.income_text or str(ext.income_amount)
        else:
            rejected.append("monthly_income")
    return clean, rejected


def missing_fields(lead: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if not lead.get(f)]


def format_lead_summary(lead: dict, lang: str = "th", product_name: str | None = None) -> str:
    inc = lead.get("monthly_income")
    inc_s = f"{inc:,.0f} บาท/เดือน" if inc else "-"
    if lang == "en":
        inc_s = f"{inc:,.0f} THB/month" if inc else "-"
        rows = [("Name", lead.get("full_name")), ("Occupation", lead.get("occupation")),
                ("Monthly income", inc_s), ("Phone", lead.get("phone")), ("Product of interest", product_name)]
    else:
        rows = [("ชื่อ", lead.get("full_name")), ("อาชีพ", lead.get("occupation")), ("รายได้", inc_s),
                ("เบอร์ติดต่อ", lead.get("phone")), ("ผลิตภัณฑ์ที่สนใจ", product_name)]
    return "\n".join(f"- {k}: {v or '-'}" for k, v in rows)
