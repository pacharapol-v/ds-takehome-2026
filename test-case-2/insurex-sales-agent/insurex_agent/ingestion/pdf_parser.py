"""Layout-aware parser for the InsureX product brochures.

The brochures mix three layouts on the same page:
  1. full-width paragraphs,
  2. 2-4 column "feature boxes" (icon + heading + 3-4 short lines),
  3. benefit / premium / underwriting tables (label column + value columns).

Generic converters either interleave the feature-box columns line by line or
split Thai combining marks into the wrong table cell (we measured this - see
README "Parsing"). This parser works on PyMuPDF line boxes:

  * cluster lines into visual rows by vertical overlap,
  * classify each row:  SINGLE (one segment), TABLE (label + values) or
    MULTI (several independent text columns stacked tightly = feature boxes),
  * MULTI regions are emitted column by column (reading order),
  * TABLE rows are emitted as markdown table rows, keeping label/value pairs,
  * headings are detected from font size + weight.

Every page is checked with `thai_quality_issues`; the result is stored in the
page metadata so corrupted extraction is visible instead of silent.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .thai_text import normalize_spaces, repair_thai, thai_quality_issues

# --- tunables (points; A4 page is 595 x 842) ---------------------------------
SEGMENT_JOIN_GAP = 6.0      # lines on the same row closer than this are one segment
TIGHT_GAP = 2.0             # rows closer than this belong to the same paragraph block
COLUMN_TOLERANCE = 45.0     # x0 distance for assigning a segment to a column
FOOTNOTE_SIZE = 7.8         # text smaller than this is legal/footnote text


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float
    bold: bool

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


@dataclass
class Segment:
    lines: list[Line]

    @property
    def x0(self): return min(l.x0 for l in self.lines)
    @property
    def x1(self): return max(l.x1 for l in self.lines)
    @property
    def y0(self): return min(l.y0 for l in self.lines)
    @property
    def y1(self): return max(l.y1 for l in self.lines)
    @property
    def text(self): return " ".join(l.text for l in self.lines).strip()
    @property
    def size(self): return max(l.size for l in self.lines)
    @property
    def bold(self): return any(l.bold for l in self.lines)


@dataclass
class Row:
    segments: list[Segment]
    kind: str = ""  # SINGLE | TABLE | MULTI

    @property
    def y0(self): return min(s.y0 for s in self.segments)
    @property
    def y1(self): return max(s.y1 for s in self.segments)


@dataclass
class ParsedPage:
    page_number: int           # 1-based
    markdown: str
    quality_issues: int
    n_table_rows: int


@dataclass
class ParsedDocument:
    file_name: str
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        return "\n\n".join(f"<!-- page {p.page_number} -->\n{p.markdown}" for p in self.pages)


# -----------------------------------------------------------------------------
def _is_numeric_value(text: str) -> bool:
    t = text.replace(" ", "")
    if not t:
        return False
    if re.fullmatch(r"แผน\d+|ชาย|หญิง|-|–|ไม่มี|จ่ายตามจริง", t):
        return True
    digits = sum(c.isdigit() for c in t)
    return digits / len(t) >= 0.35


def _extract_lines(page: pymupdf.Page) -> list[Line]:
    lines: list[Line] = []
    data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for ln in block["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = repair_thai("".join(s["text"] for s in ln["spans"]))
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            size = max(s["size"] for s in spans)
            bold = any(("Bold" in s["font"] or "SemiBold" in s["font"]) for s in spans)
            lines.append(Line(x0, y0, x1, y1, text, size, bold))
    return lines


def _group_rows(lines: list[Line]) -> list[Row]:
    """Cluster lines into visual rows, then merge horizontally adjacent lines."""
    rows: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: (l.yc, l.x0)):
        if rows:
            cur = rows[-1]
            mean_yc = statistics.fmean(l.yc for l in cur)
            ref_h = min(statistics.fmean(l.h for l in cur), ln.h)
            if abs(ln.yc - mean_yc) < 0.5 * ref_h:
                cur.append(ln)
                continue
        rows.append([ln])

    out: list[Row] = []
    for row_lines in rows:
        row_lines.sort(key=lambda l: l.x0)
        segs: list[Segment] = []
        for ln in row_lines:
            if segs and ln.x0 - segs[-1].x1 < SEGMENT_JOIN_GAP:
                segs[-1].lines.append(ln)
            else:
                segs.append(Segment([ln]))
        out.append(Row(segs))
    return out


def _merge_wrapped_labels(rows: list[Row]) -> list[Row]:
    """A table label wrapped onto 2 lines produces a label-only row that overlaps
    vertically with the value row. Merge it into the neighbouring table row."""
    merged: list[Row] = []
    for row in rows:
        if (
            merged
            and len(row.segments) == 1
            and len(merged[-1].segments) >= 2
            and any(_is_numeric_value(s.text) for s in merged[-1].segments[1:])
            and row.y0 < merged[-1].y1 - 1.0                      # vertical overlap = wrapped line
            and abs(row.segments[0].x0 - merged[-1].segments[0].x0) < 8
        ):
            merged[-1].segments[0].lines.append(row.segments[0].lines[0])
            continue
        merged.append(row)
    return merged


def _classify(rows: list[Row]) -> None:
    """MULTI = part of a tightly stacked block of >= 2 columns (feature boxes).
    TABLE = an isolated label/value row. Tightness, not content, decides: table
    rows are separated by cell padding, feature-box lines are typeset as one
    paragraph per column with (almost) no gap."""
    def multi(i): return 0 <= i < len(rows) and len(rows[i].segments) >= 2
    def gap(a, b): return rows[b].y0 - rows[a].y1

    for i, row in enumerate(rows):
        if len(row.segments) == 1:
            row.kind = "SINGLE"
            continue
        tight = (multi(i - 1) and gap(i - 1, i) < TIGHT_GAP) or (multi(i + 1) and gap(i, i + 1) < TIGHT_GAP)
        row.kind = "MULTI" if tight else "TABLE"
    # a tight block whose rows mostly carry numeric values is a dense table
    # (e.g. a two-line column header glued to the first data row), not boxes.
    i = 0
    while i < len(rows):
        if rows[i].kind != "MULTI":
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and rows[j + 1].kind == "MULTI":
            j += 1
        block = rows[i:j + 1]
        numeric_rows = sum(any(_is_numeric_value(s.text) for s in r.segments[1:]) for r in block)
        if numeric_rows / len(block) >= 0.5:
            for r in block:
                r.kind = "TABLE"
        i = j + 1
    # a row of column headings sitting just above a MULTI block belongs to it
    for i, row in enumerate(rows[:-1]):
        nxt = rows[i + 1]
        if (row.kind == "TABLE" and nxt.kind == "MULTI" and nxt.y0 - row.y1 < 10
                and all(s.bold and s.size >= 12.5 for s in row.segments)):
            row.kind = "MULTI"


def _heading_level(seg: Segment, body_size: float) -> int:
    """0 = not a heading; 1/2/3 = markdown heading level."""
    if seg.size >= body_size * 2.0 and seg.bold:
        return 1
    if seg.size >= max(13.0, body_size * 1.2) and seg.bold and len(seg.text) <= 80:
        return 2
    return 0


def _render(rows: list[Row], page_width: float, small_layer: bool = False) -> tuple[str, int]:
    sizes = [l.size for r in rows for s in r.segments for l in s.lines]
    body_size = statistics.median(sizes) if sizes else 10.0
    out: list[str] = []
    n_table = 0
    region: list[list[Segment]] = []      # columns of the current MULTI region
    region_x: list[float] = []
    region_y1 = 0.0

    def fmt(seg: Segment) -> str:
        lvl = _heading_level(seg, body_size)
        if lvl:
            return f"\n{'#' * (lvl + 1)} {seg.text}\n"
        if seg.size < FOOTNOTE_SIZE:
            return f"<small>{seg.text}</small>"
        return seg.text

    def flush():
        nonlocal region, region_x
        for col in region:
            out.append("\n".join(fmt(s) for s in col))
            out.append("")
        region, region_x = [], []

    def add_to_region(seg: Segment):
        for idx, x in enumerate(region_x):
            if abs(seg.x0 - x) < COLUMN_TOLERANCE or (seg.x0 < x + 120 and seg.x1 > x and abs((seg.x0 + seg.x1) / 2 - (region[idx][0].x0 + region[idx][0].x1) / 2) < 60):
                region[idx].append(seg)
                return
        region_x.append(seg.x0)
        region.append([seg])
        order = sorted(range(len(region_x)), key=lambda k: region_x[k])
        region[:] = [region[k] for k in order]
        region_x[:] = [region_x[k] for k in order]

    for row in rows:
        if row.kind == "MULTI":
            for seg in row.segments:
                add_to_region(seg)
            region_y1 = row.y1
            continue
        if (
            row.kind == "SINGLE" and region
            and row.y0 - region_y1 < TIGHT_GAP + 2
            and (row.segments[0].x1 - row.segments[0].x0) < page_width * 0.55
        ):
            add_to_region(row.segments[0])          # extra line of one feature-box column
            region_y1 = row.y1
            continue
        flush()
        if small_layer:                              # small print: plain lines, no tables
            for seg in row.segments:
                if len(seg.text) > 4:                # drop icon labels such as "TAX"
                    out.append(f"<small>{seg.text}</small>")
            continue
        if row.kind == "SINGLE":
            out.append(fmt(row.segments[0]))
        else:  # TABLE
            cells = [s.text.replace("|", "/") for s in row.segments]
            out.append("| " + " | ".join(cells) + " |")
            n_table += 1
    flush()

    # join consecutive plain lines into paragraphs, keep table rows / headings separate
    text = "\n".join(out)
    text = re.sub(r"\n(?!\||#|\n|<small>)(?<!\|\n)", "\n", text)
    return normalize_spaces(text), n_table


def _page_markdown(lines: list[Line], page_width: float, small_layer: bool = False) -> tuple[str, int]:
    rows = _merge_wrapped_labels(_group_rows(lines))
    _classify(rows)
    return _render(rows, page_width, small_layer)


def parse_pdf(path: str | Path) -> ParsedDocument:
    path = Path(path)
    doc = pymupdf.open(path)
    parsed = ParsedDocument(file_name=path.name)
    for i, page in enumerate(doc):
        lines = _extract_lines(page)
        # small print is laid out independently of the body (side notes, legal
        # footer); processing it as its own layer keeps it from being
        # interleaved with feature boxes / tables.
        body = [l for l in lines if l.size >= FOOTNOTE_SIZE]
        small = [l for l in lines if l.size < FOOTNOTE_SIZE]
        md_body, n_table = _page_markdown(body, page.rect.width)
        md_small, _ = _page_markdown(small, page.rect.width, small_layer=True) if small else ("", 0)
        md = md_body + (f"\n\n{md_small}" if md_small else "")
        parsed.pages.append(
            ParsedPage(page_number=i + 1, markdown=md, quality_issues=thai_quality_issues(md), n_table_rows=n_table)
        )
    return parsed
