"""Product master data for the knowledge base.

In production this would come from the product master table; here it is a small
registry so that (1) every chunk carries clean product metadata, (2) the agent
can detect which product a question is about and filter retrieval, and
(3) the lead record stores a canonical product id instead of free text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Product:
    product_id: str
    file_name: str
    name_th: str             # marketing name used in the brochure
    name_en: str
    official_plan_th: str    # registered plan name (from the brochure footnote)
    product_type: str        # whole_life | whole_life_participating | term
    summary_th: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


PRODUCTS: tuple[Product, ...] = (
    Product(
        product_id="khum_cheeva_90_10",
        file_name="khum_cheeva_90_10.pdf",
        name_th="คุ้มชีวา 90/10",
        name_en="Khum Cheeva 90/10",
        official_plan_th="แบบตลอดชีพ 90/10 แนบสัญญาเพิ่มเติมตามเงื่อนไข",
        product_type="whole_life",
        summary_th="ประกันชีวิตตลอดชีพคุ้มครองถึงอายุ 90 ปี ชำระเบี้ย 10 ปี แนบความคุ้มครองอุบัติเหตุ โรคร้ายแรง และมะเร็ง",
        aliases=("คุ้มชีวา", "ชีวา", "90/10", "khum cheeva", "cheeva"),
    ),
    Product(
        product_id="khum_cheevee_extra_99",
        file_name="khum_cheevee_extra_99.pdf",
        name_th="คุ้มชีวี เอ็กซ์ตร้า 99/10, 99/15, 99/20",
        name_en="Khum Cheevee Extra 99/10, 99/15, 99/20",
        official_plan_th="แบบตลอดชีพพิเศษ ชำระเบี้ย 10/15/20 ปี (ชนิดมีเงินปันผล) แนบสัญญาเพิ่มเติมตามเงื่อนไข",
        product_type="whole_life_participating",
        summary_th="ประกันชีวิตตลอดชีพถึงอายุ 99 ปี มีเงินคืนทุกปี 1% และเงินปันผล เลือกชำระเบี้ย 10, 15 หรือ 20 ปี",
        aliases=("คุ้มชีวี", "ชีวี", "เอ็กซ์ตร้า", "เอ็กตร้า", "extra", "cheevee", "99/10", "99/15", "99/20"),
    ),
    Product(
        product_id="whole_life_99_99",
        file_name="whole_life_99_99.pdf",
        name_th="ตลอดชีพ 99/99",
        name_en="Whole Life 99/99",
        official_plan_th="แบบประกันตลอดชีพ 99/99",
        product_type="whole_life",
        summary_th="ประกันชีวิตตลอดชีพคุ้มครองและชำระเบี้ยถึงอายุ 99 ปี ครบสัญญารับ 101% ของทุนประกัน",
        aliases=("99/99", "ตลอดชีพ 99/99", "whole life 99/99", "wl 99/99"),
    ),
    Product(
        product_id="whole_life_90_20",
        file_name="whole_life_90_20.pdf",
        name_th="ตลอดชีพ 90/20",
        name_en="Whole Life 90/20",
        official_plan_th="แบบตลอดชีพ 90/20",
        product_type="whole_life",
        summary_th="ประกันชีวิตตลอดชีพคุ้มครองถึงอายุ 90 ปี ชำระเบี้ย 20 ปี ครบสัญญารับ 100% ของทุนประกัน",
        aliases=("90/20", "ตลอดชีพ 90/20", "whole life 90/20", "wl 90/20"),
    ),
    Product(
        product_id="khum_manjai_sure_term",
        file_name="khum_manjai_sure_term.pdf",
        name_th="คุ้มมั่นใจชัวร์ แบบชั่วระยะเวลา (5/5, 10/10, 15/15)",
        name_en="Khum Manjai Sure Term 5/5, 10/10, 15/15",
        official_plan_th="แบบกำหนดระยะเวลา 5/5 10/10 และ 15/15",
        product_type="term",
        summary_th="ประกันชีวิตแบบชั่วระยะเวลา คุ้มครอง 5, 10 หรือ 15 ปี ทุนประกันสูง เบี้ยต่ำ เบี้ยคงที่ตามอายุแรกเข้า",
        aliases=("คุ้มมั่นใจชัวร์", "มั่นใจชัวร์", "คุ้มมั่นใจ", "ชั่วระยะเวลา", "กำหนดระยะเวลา", "term", "term life",
                 "5/5", "10/10", "15/15"),
    ),
)

PRODUCT_BY_ID = {p.product_id: p for p in PRODUCTS}
PRODUCT_BY_FILE = {p.file_name: p for p in PRODUCTS}

# Generic category words -> product ids (used when no specific product is named)
CATEGORY_HINTS = {
    "ตลอดชีพ": ("khum_cheeva_90_10", "khum_cheevee_extra_99", "whole_life_99_99", "whole_life_90_20"),
    "whole life": ("khum_cheeva_90_10", "khum_cheevee_extra_99", "whole_life_99_99", "whole_life_90_20"),
    "เงินปันผล": ("khum_cheevee_extra_99",),
    "เงินคืนทุกปี": ("khum_cheevee_extra_99",),
}

# Words added to the Thai tokenizer dictionary so product names stay intact.
CUSTOM_DICTIONARY = (
    "คุ้มชีวา", "คุ้มชีวี", "เอ็กซ์ตร้า", "คุ้มมั่นใจชัวร์", "ชั่วระยะเวลา", "ตลอดชีพ", "ตลอดชีพพิเศษ",
    "เอดีดีพลัส", "เอดีดีอาร์ซีซีพลัส", "เอไอ", "ซีไอ", "อินชัวร์", "เอกซ์", "เอฟดับบลิวดี",
)

# Customer wording -> brochure wording (query-side expansion for lexical retrieval)
SYNONYMS: dict[str, str] = {
    "ทุนประกัน": "จำนวนเงินเอาประกันภัย", "ทุน": "จำนวนเงินเอาประกันภัย", "sum assured": "จำนวนเงินเอาประกันภัย",
    "ค่าเบี้ย": "เบี้ยประกันภัย", "ราคา": "เบี้ยประกันภัย", "เบี้ย": "เบี้ยประกันภัย", "premium": "เบี้ยประกันภัย",
    "ผู้หญิง": "เพศหญิง", "ผู้ชาย": "เพศชาย", "female": "เพศหญิง", "male": "เพศชาย",
    "ตาย": "เสียชีวิต", "death": "เสียชีวิต",
    "รับประกันอายุ": "อายุรับประกันภัย", "รับอายุ": "อายุรับประกันภัย", "อายุเท่าไหร่ถึงทำได้": "อายุรับประกันภัย",
    "จ่ายกี่ปี": "ระยะเวลาชำระเบี้ยประกันภัย", "เบี้ยกี่ปี": "ระยะเวลาชำระเบี้ยประกันภัย",
    "ชำระกี่ปี": "ระยะเวลาชำระเบี้ยประกันภัย", "ผ่อนกี่ปี": "ระยะเวลาชำระเบี้ยประกันภัย",
    "คุ้มครองกี่ปี": "ระยะเวลาคุ้มครอง ระยะเวลาเอาประกันภัย", "คุ้มครองถึง": "ระยะเวลาเอาประกันภัย",
    "ครบสัญญา": "ครบกำหนดสัญญา", "ปันผล": "เงินปันผล", "dividend": "เงินปันผล",
    "ผู้รับประกัน": "รับประกันภัยโดย บริษัท", "บริษัทไหน": "รับประกันภัยโดย บริษัท", "insurer": "รับประกันภัยโดย",
    "ภาษี": "ลดหย่อนภาษี", "tax": "ลดหย่อนภาษี",
    "จ่ายรายเดือน": "รายเดือน งวดการชำระเบี้ยประกันภัย", "ผ่อนรายเดือน": "รายเดือน งวดการชำระเบี้ยประกันภัย",
    "มะเร็ง": "โรคมะเร็ง", "cancer": "โรคมะเร็ง",
}


def expand_query(query: str) -> str:
    """Append brochure vocabulary for colloquial terms found in the query."""
    low = query.lower()
    extra = [v for k, v in SYNONYMS.items() if k in low and v not in query]
    return query + (" " + " ".join(dict.fromkeys(extra)) if extra else "")


def _alias_pattern(alias: str) -> re.Pattern:
    # product codes must not match inside longer numbers ("90/10" must not hit "190/100")
    if re.fullmatch(r"\d+/\d+", alias):
        return re.compile(rf"(?<![\d/]){re.escape(alias)}(?![\d/])")
    return re.compile(re.escape(alias), re.IGNORECASE)


_ALIAS_PATTERNS = [(p.product_id, _alias_pattern(a)) for p in PRODUCTS for a in p.aliases]


def detect_products(text: str) -> list[str]:
    """Return product ids explicitly mentioned in the text (ordered by first mention)."""
    hits: dict[str, int] = {}
    for pid, pat in _ALIAS_PATTERNS:
        m = pat.search(text or "")
        if m and (pid not in hits or m.start() < hits[pid]):
            hits[pid] = m.start()
    return sorted(hits, key=hits.get)


def catalog_overview_th() -> str:
    """Short catalogue used by the agent for 'what products do you have?' questions."""
    lines = [f"- {p.name_th}: {p.summary_th}" for p in PRODUCTS]
    return "\n".join(lines)


def catalog_overview_en() -> str:
    return "\n".join(f"- {p.name_en}" for p in PRODUCTS)


# insurance types InsureX's life catalogue does not cover; such a question is never a catalogue question
OTHER_INSURANCE_RE = re.compile(r"ประกันรถ|รถยนต์|รถจักรยานยนต์|มอเตอร์ไซค์|เดินทาง|สุขภาพ|ประกันบ้าน|อัคคีภัย|"
                                r"ทรัพย์สิน|สัตว์เลี้ยง|\b(?:car|motor|auto|travel|health|home|house|fire|pet)\b",
                                re.I)   # whole terms: a bare "รถ" would also match inside "สามารถ"

# plan codes that exist ("90/10", "99/15", "5/5", ...), taken from the aliases so the catalogue stays the one source
KNOWN_PLAN_CODES = frozenset(a for p in PRODUCTS for a in p.aliases if re.fullmatch(r"\d+/\d+", a))
_PLAN_CODE_RE = re.compile(r"(?<![\d/.,])(\d{1,2}/\d{1,2})(?![\d/])")   # not dates like 12/11/2026


def unknown_plan_codes(text: str) -> list[str]:
    """Plan codes in the text that no product has, e.g. "คุ้มชีวา 99/50" -> ["99/50"]."""
    return [c for c in dict.fromkeys(_PLAN_CODE_RE.findall(text or "")) if c not in KNOWN_PLAN_CODES]
