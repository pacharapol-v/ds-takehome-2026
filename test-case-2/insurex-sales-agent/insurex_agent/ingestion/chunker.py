"""Turn parsed brochure markdown into a parent/child chunk hierarchy.

Retrieval unit (child)                     Generation unit (parent)
--------------------------------------    ------------------------------------
* one table row rewritten as a fact,       * the whole section it came from
  e.g. "คุ้มชีวา 90/10 > ตัวอย่างผลประโยชน์   (full table / full feature list),
  > กรณีเสียชีวิตจากอุบัติเหตุ >              so the LLM sees column headers,
  เสียชีวิตจากอุบัติเหตุสาธารณะ: แผน 1 =       footnotes and neighbouring rows.
  1,200,000 | แผน 2 = 2,000,000 ..."
* one feature box / paragraph
* one legal bullet

Small, precise children give sharp similarity scores (a premium question hits
the premium row, not the marketing paragraph); parents give the LLM enough
context to answer correctly. Every child text starts with a *contextual header*
(product name + section) because a bare row such as "แผน 1 = 300,000" is
meaningless and identical across products.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .catalog import PRODUCT_BY_FILE, Product
from .pdf_parser import ParsedDocument

SECTION_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^ตัวอย่างเบี้ยประกันภัย"), "premium_example"),
    (re.compile(r"^ตัวอย่างผลประโยชน์|^ผลประโยชน์ความคุ้มครอง\s*\(ตัวอย่าง"), "benefit_example"),
    (re.compile(r"^ผลประโยชน์ความคุ้มครอง"), "benefits"),
    (re.compile(r"^เงื่อนไขการรับประกันภัย"), "underwriting"),
    (re.compile(r"^หมายเหตุ$"), "notes_legal"),
]
SECTION_LABEL_TH = {
    "overview": "ภาพรวมและจุดเด่นของแบบประกัน",
    "benefits": "ผลประโยชน์ความคุ้มครอง",
    "benefit_example": "ตัวอย่างผลประโยชน์ความคุ้มครอง",
    "premium_example": "ตัวอย่างเบี้ยประกันภัย",
    "underwriting": "เงื่อนไขการรับประกันภัย",
    "notes_legal": "หมายเหตุ ข้อกำหนด และข้อมูลบริษัท",
    "product_card": "ข้อมูลสรุปผลิตภัณฑ์",
}
# brochure-specific: lines that continue the previous table row (wrapped cell text)
CONTINUATION_RE = re.compile(r"^(จ่ายตามจริง|ชำระเบี้ยประกันภัย$|\(|อายุ \d|หรือมูลค่า|แล้วแต่|หรือ\s)")
HEADER_CELL_RE = re.compile(r"^(แผน\s*\d+|ชาย|หญิง|\d{1,2}/\d{1,2}|คุ้ม.*|แบบ.*|หน่วย.*|ตลอดชีพ.*)$")
TABLE_SECTION_ROW_RE = re.compile(r"^\|\s*(ตัวอย่าง[^|]*|ผลประโยชน์ความคุ้มครอง[^|]*)\|")


@dataclass
class Chunk:
    chunk_id: str
    text: str                     # what gets embedded / BM25-indexed
    parent_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Parent:
    parent_id: str
    text: str                     # what the LLM reads
    metadata: dict = field(default_factory=dict)


@dataclass
class _Section:
    kind: str
    title: str
    page: int
    lines: list[str] = field(default_factory=list)


def _cid(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _is_header(cells: list[str]) -> bool:
    values = [c for c in cells[1:] if c] or cells
    return all(HEADER_CELL_RE.match(c) for c in values)


def _section_kind(text: str) -> str | None:
    t = text.strip().lstrip("#").strip()
    for pat, kind in SECTION_RULES:
        if pat.search(t):
            return kind
    return None


def _split_sections(parsed: ParsedDocument) -> tuple[list[_Section], list[tuple[int, str]]]:
    sections: list[_Section] = []
    legal: list[tuple[int, str]] = []
    for page in parsed.pages:
        cur = _Section("overview" if page.page_number == 1 else "benefits", "", page.page_number)
        sections.append(cur)
        for raw in page.markdown.splitlines():
            line = raw.strip()
            if not line:
                cur.lines.append("")
                continue
            if line.startswith("<small>"):
                legal.append((page.page_number, re.sub(r"</?small>", "", line)))
                continue
            title_src = line
            m = TABLE_SECTION_ROW_RE.match(line)
            if m:
                title_src = m.group(1)
            kind = None if (line.startswith("|") and not m) else _section_kind(title_src)
            if kind and page.page_number > 1:
                if kind == "notes_legal":  # "หมายเหตุ" block on page 3 of the term brochure
                    cur = _Section(kind, "หมายเหตุ", page.page_number)
                else:
                    cur = _Section(kind, title_src.lstrip("#").strip(), page.page_number)
                sections.append(cur)
                continue
            cur.lines.append(line)
    # drop empty sections
    return [s for s in sections if any(l.strip() for l in s.lines)], legal


def _row_fact(product: Product, section: _Section, path: list[str], header: list[str] | None, cells: list[str],
              plan_sa: dict[str, str] | None = None) -> str:
    label, values = cells[0], cells[1:]
    if header:
        cols = [c for c in header if c]
        # right-align header to value columns ("| ชาย | หญิง |" over "| 20 | 3,195 | 2,700 |")
        cols = cols[-len(values):] if len(cols) >= len(values) else cols
        if len(cols) == len(values) and len(values) >= 2:
            if plan_sa and not label.startswith("จำนวนเงินเอาประกันภัย"):
                # premium rows only say "แผน 3"; make the sum assured explicit
                cols = [f"{c} (ทุนประกัน {plan_sa[c]} บาท)" if c in plan_sa else c for c in cols]
            values = [f"{c} = {v}" for c, v in zip(cols, values)]
    title = section.title or SECTION_LABEL_TH[section.kind]
    prefix = " > ".join([product.name_th, title, *path])
    return f"{prefix} > {label}: " + " | ".join(values)


def _blocks(lines: list[str]) -> list[list[str]]:
    blocks, cur = [], []
    for l in lines:
        if not l.strip():
            if cur:
                blocks.append(cur)
            cur = []
        else:
            cur.append(l)
    if cur:
        blocks.append(cur)
    return blocks


def build_chunks(parsed: ParsedDocument) -> tuple[list[Parent], list[Chunk]]:
    product = PRODUCT_BY_FILE[parsed.file_name]
    base_meta = {
        "product_id": product.product_id,
        "product_name": product.name_th,
        "product_type": product.product_type,
        "source": parsed.file_name,
    }
    parents: list[Parent] = []
    children: list[Chunk] = []
    sections, legal = _split_sections(parsed)

    header: list[str] | None = None
    plan_sa: dict[str, str] = {}          # "แผน 1" -> "300,000" (sum assured per example plan)
    prev: _Section | None = None
    for sec_idx, sec in enumerate(sections):
        if prev is None or prev.page != sec.page:
            plan_sa = {}
        # column headers only carry over between example tables on the same page
        if not (prev and prev.page == sec.page and prev.kind in ("benefit_example", "premium_example")
                and sec.kind in ("benefit_example", "premium_example")):
            header = None
        prev = sec
        title = sec.title or SECTION_LABEL_TH[sec.kind]
        pid = _cid(product.product_id, sec.page, sec_idx, title)
        body = "\n".join(sec.lines).strip()
        # carry the active column header into a section that starts with data rows
        first_row = next((l for l in sec.lines if l.startswith("|")), None)
        if header and first_row and not _is_header(_cells(first_row)) and len(_cells(first_row)) > 2:
            body = "| " + " | ".join(["", *header] if len(header) < len(_cells(first_row)) else header) + " |\n" + body
        if plan_sa and sec.kind == "premium_example":
            body = ("(จำนวนเงินเอาประกันภัยของแต่ละแผน: " +
                    " | ".join(f"{k} = {v} บาท" for k, v in plan_sa.items()) + ")\n" + body)
        meta = {**base_meta, "page": sec.page, "section": sec.kind, "section_title": title}
        parents.append(Parent(pid, f"[{product.name_th} | {title} | หน้า {sec.page}]\n{body}", meta))

        path: list[str] = []
        prev_kind = "start"
        rows_in_section = any(l.startswith("|") for l in sec.lines)
        n = 0
        if rows_in_section and sec.kind != "overview":
            lines = [l for l in sec.lines if l.strip()]
            for i, line in enumerate(lines):
                if line.startswith("|"):
                    cells = _cells(line)
                    if cells[0].startswith("InsureX Thailand"):   # social-media footer
                        continue
                    if _is_header(cells) and len(cells) > 1:
                        header = cells if prev_kind != "header" else _merge_headers(header, cells)
                        prev_kind = "header"
                        continue
                    use_path = path if cells[0].startswith("-") else path[:1]
                    if (cells[0].startswith("จำนวนเงินเอาประกันภัย") and header and not plan_sa
                            and any(h.startswith("แผน") for h in header)):
                        cols = [c for c in header if c][-len(cells[1:]):]
                        plan_sa = {c: v for c, v in zip(cols, cells[1:]) if c.startswith("แผน")}
                    fact = _row_fact(product, sec, use_path, header, cells, plan_sa)
                    children.append(Chunk(_cid(pid, "row", n), fact, pid,
                                          {**meta, "chunk_type": "table_row"}))
                    n += 1
                    prev_kind = "row"
                    # a multi-value row with no header above it acts as the header
                    # for the rows below ("ระยะเวลาชำระเบี้ย: 5 ปี | 10 ปี | 15 ปี")
                    if header is None and len(cells) >= 3:
                        header = [f"{cells[0]} {v}" for v in cells[1:]]
                else:
                    text = line.lstrip("#").strip()
                    if prev_kind == "row" and CONTINUATION_RE.match(text) and children:
                        children[-1].text += f" ({text.strip('()')})"
                        continue
                    path = [text] if prev_kind != "label" else (path + [text])[-2:]
                    prev_kind = "label"
            # also index the whole section once (helps "list everything" questions)
            children.append(Chunk(_cid(pid, "section"), f"{product.name_th} > {title}\n{body}"[:1500], pid,
                                  {**meta, "chunk_type": "section"}))
        else:
            for b_idx, block in enumerate(_blocks(sec.lines)):
                text = " ".join(l.lstrip("#").strip() for l in block)
                if len(text) < 15:
                    continue
                children.append(Chunk(_cid(pid, "para", b_idx), f"{product.name_th} > {title} > {text}", pid,
                                      {**meta, "chunk_type": "paragraph"}))

    # legal / footnote text: one parent, one child per bullet
    if legal:
        pid = _cid(product.product_id, "legal")
        joined = " ".join(t for _, t in legal)
        meta = {**base_meta, "page": legal[-1][0], "section": "notes_legal",
                "section_title": SECTION_LABEL_TH["notes_legal"]}
        parents.append(Parent(pid, f"[{product.name_th} | {SECTION_LABEL_TH['notes_legal']}]\n{joined}", meta))
        bullets = [b.strip() for b in re.split(r"•", joined) if len(b.strip()) > 15]
        for b_idx, b in enumerate(bullets):
            children.append(Chunk(_cid(pid, b_idx), f"{product.name_th} > หมายเหตุ > {b}", pid,
                                  {**meta, "chunk_type": "legal"}))

    # product card (master data) - answers "what is X / what products exist"
    pid = _cid(product.product_id, "card")
    card = (f"ชื่อผลิตภัณฑ์: {product.name_th} ({product.name_en})\n"
            f"ชื่อแบบประกันตามกรมธรรม์: {product.official_plan_th}\n"
            f"ประเภท: {product.product_type}\nสรุป: {product.summary_th}\n"
            f"ชื่ออื่นที่ลูกค้าอาจเรียก: {', '.join(product.aliases)}")
    meta = {**base_meta, "page": 1, "section": "product_card", "section_title": SECTION_LABEL_TH["product_card"]}
    parents.append(Parent(pid, f"[{product.name_th} | ข้อมูลสรุปผลิตภัณฑ์]\n{card}", meta))
    children.append(Chunk(_cid(pid, 0), card, pid, {**meta, "chunk_type": "product_card"}))
    return parents, children


def _merge_headers(prev: list[str] | None, cur: list[str]) -> list[str]:
    """Two stacked header rows ("คุ้มชีวี เอ็กซ์ตร้า" / "99/10") -> keep the more specific one."""
    if not prev:
        return cur
    return cur if len([c for c in cur if c]) >= len([c for c in prev if c]) - 1 else prev


def save_processed(parsed: ParsedDocument, parents: list[Parent], children: list[Chunk], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(parsed.file_name).stem
    (out_dir / f"{stem}.md").write_text(parsed.markdown, encoding="utf-8")
    with open(out_dir / f"{stem}.chunks.jsonl", "w", encoding="utf-8") as f:
        for c in children:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
