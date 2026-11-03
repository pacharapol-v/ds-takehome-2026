"""Thai text utilities: PDF glyph repair, normalisation, tokenisation, amount parsing.

Why this module exists
----------------------
The InsureX brochures are exported from Adobe Illustrator with the SukhumvitSet
font. Thai tone marks and some vowels are emitted as *Private Use Area* (PUA)
glyphs (U+F700-U+F71A) - "shifted" variants the font uses for positioning.
A naive extractor either drops them or keeps invisible PUA characters, so
"คุ้มครอง" becomes "คุมครอง" and keyword search / embeddings degrade badly.
`repair_thai` maps them back to standard Unicode.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Standard Thai PUA -> Unicode mapping (Microsoft/Apple Thai font convention).
_PUA_TARGETS = (
    "ฐิีึื่้๊๋์"  # F700-F709
    "่้๊๋์ญัํ็่"  # F70A-F713
    "้๊๋์ฺุู"                    # F714-F71A
)
PUA_MAP = {0xF700 + i: ch for i, ch in enumerate(_PUA_TARGETS)}

# Combining marks that must sit on a base character.
_UPPER_VOWELS = "ัิีึื็"
_LOWER_VOWELS = "ฺุู"
_TONES = "่้๊๋์ํ๎"
COMBINING = _UPPER_VOWELS + _LOWER_VOWELS + _TONES
_COMBINING_RE = f"[{COMBINING}]"
# A combining mark directly after whitespace / start / punctuation is orphaned.
_ORPHAN_RE = re.compile(rf"(?:(?<=^)|(?<=[\s|>(\[]))({_COMBINING_RE}+)")
# Two vowel marks stacked on one consonant never happen in valid Thai.
_INVALID_STACK_RE = re.compile(rf"[{_UPPER_VOWELS}{_LOWER_VOWELS}]{{2,}}")


def repair_thai(text: str) -> str:
    """Fix PUA glyphs, decomposed SARA AM and stray zero-width characters."""
    text = text.translate(PUA_MAP)
    # NIKHAHIT (ํ) + optional tone + SARA AA (า)  ->  tone + SARA AM (ำ)
    text = re.sub("ํ([่-๋]?)า", lambda m: m.group(1) + "ำ", text)
    text = text.replace("​", "").replace("﻿", "").replace(" ", " ")
    text = unicodedata.normalize("NFC", text)
    return text


def thai_quality_issues(text: str) -> int:
    """Count signs of glyph-level corruption (used as an extraction quality gate)."""
    return len(_ORPHAN_RE.findall(text)) + len(_INVALID_STACK_RE.findall(text))


def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Tokenisation (for BM25 / lexical matching)
# ---------------------------------------------------------------------------
# Product codes such as 90/10, 99/99, 5/5 must stay one token, otherwise
# "90/10" and "90/20" collapse into overlapping tokens.
_CODE_RE = re.compile(r"(\d{1,2})\s*/\s*(\d{1,2})")
_STOP = {
    "ที่", "และ", "หรือ", "ของ", "ใน", "การ", "ได้", "ให้", "เป็น", "มี", "จะ", "ไม่", "กับ",
    "แล้ว", "ครับ", "ค่ะ", "คะ", "นะ", "จ้า", "อะไร", "ไหม", "มั้ย", "บ้าง", "หน่อย", "อยาก",
    "ทราบ", "ขอ", "ถาม", "the", "a", "an", "of", "is", "what", "for", "to", "and", "in",
}


@lru_cache(maxsize=1)
def _tokenizer():
    from pythainlp.tokenize import Tokenizer
    from pythainlp.corpus.common import thai_words

    from .catalog import CUSTOM_DICTIONARY

    words = set(thai_words()) | set(CUSTOM_DICTIONARY)
    return Tokenizer(custom_dict=words, engine="newmm", keep_whitespace=False)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Thai+English word tokenisation tuned for insurance product text."""
    text = repair_thai(text).lower()
    codes = [f"code{a}x{b}" for a, b in _CODE_RE.findall(text)]   # keep 90/10 as ONE token
    text = _CODE_RE.sub(" ", text)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,000,000 -> 1000000
    tokens = _tokenizer().word_tokenize(text) + codes
    out = []
    for tok in tokens:
        tok = tok.strip()
        if not tok or re.fullmatch(r"[\W_]+", tok):
            continue
        if remove_stopwords and tok in _STOP:
            continue
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Amount parsing ("3 หมื่น", "35k", "สามหมื่นห้า", "1.2 แสน")
# ---------------------------------------------------------------------------
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_UNIT = {"สิบ": 10, "ร้อย": 100, "พัน": 1_000, "หมื่น": 10_000, "แสน": 100_000, "ล้าน": 1_000_000,
         "k": 1_000, "K": 1_000, "m": 1_000_000, "M": 1_000_000}
_NUMWORD = {"ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1, "สอง": 2, "ยี่": 2, "สาม": 3, "สี่": 4, "ห้า": 5,
            "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9}


def parse_thai_amount(text: str) -> float | None:
    """Parse a money amount written in Thai/English shorthand. Returns baht or None.

    Handles: "50,000", "50000 บาท", "5 หมื่น", "3.5 หมื่น", "35k", "1.2 แสน",
    "สามหมื่นห้า" (=35,000), "หนึ่งแสนสองหมื่น" (=120,000).
    """
    if not text:
        return None
    s = text.translate(_THAI_DIGITS).replace(",", "").strip()

    # 1) numeric with optional unit:  "3.5 หมื่น", "35k", "120000"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ล้าน|แสน|หมื่น|พัน|ร้อย|k|K|m|M)?(?![a-zA-Z])", s)
    if m:
        value = float(m.group(1))
        unit = m.group(2)
        if unit:
            value *= _UNIT[unit]
            # "3 หมื่น 5" / "3 หมื่น 5 พัน" -> trailing digit belongs to next lower unit
            rest = s[m.end():]
            m2 = re.match(r"\s*(\d+(?:\.\d+)?)\s*(แสน|หมื่น|พัน|ร้อย)?", rest)
            if m2 and unit in ("ล้าน", "แสน", "หมื่น", "พัน"):
                lower = m2.group(2)
                step = _UNIT[lower] if lower else _UNIT[unit] / 10
                value += float(m2.group(1)) * step
        return value

    # 2) spelled-out Thai: "สามหมื่นห้า", "หนึ่งแสนสองหมื่น"
    words = re.findall("|".join(sorted(list(_NUMWORD) + [u for u in _UNIT if len(u) > 1], key=len, reverse=True)), s)
    if not words:
        return None
    total, current, last_unit = 0.0, None, None
    for w in words:
        if w in _NUMWORD:
            current = _NUMWORD[w]
        else:
            unit = _UNIT[w]
            total += (current if current is not None else 1) * unit
            current, last_unit = None, unit
    if current is not None:  # trailing digit, e.g. "สามหมื่นห้า" -> 5 * (10000/10)
        total += current * (last_unit / 10 if last_unit and last_unit >= 100 else 1)
    return total or None


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Character n-grams over Thai/alnum text with whitespace removed.
    Thai is written without spaces, so word segmentation errors are a common
    retrieval failure; n-grams are segmentation-free and add recall."""
    t = re.sub(r"(?<=\d),(?=\d{3})", "", repair_thai(text).lower())
    t = re.sub(r"[^\u0e00-\u0e7fa-z0-9/%]+", "", t)
    return [t[i:i + n] for i in range(max(0, len(t) - n + 1))]
