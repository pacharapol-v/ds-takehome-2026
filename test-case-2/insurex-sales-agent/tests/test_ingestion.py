import pytest

from insurex_agent.config import ROOT
from insurex_agent.ingestion.catalog import PRODUCTS, detect_products, expand_query
from insurex_agent.ingestion.chunker import build_chunks
from insurex_agent.ingestion.pdf_parser import parse_pdf
from insurex_agent.ingestion.thai_text import (char_ngrams, parse_thai_amount, repair_thai, thai_quality_issues,
                                               tokenize)

PDF_DIR = ROOT / "data" / "pdfs"


# --------------------------------------------------------------------------- Thai text
def test_repair_thai_pua_and_sara_am():
    raw = "คุมครอง ทุมเท เปน ทํางาน นํ้า"   # PUA tone marks + decomposed sara am
    assert repair_thai(raw) == "คุ้มครอง ทุ่มเท เป็น ทำงาน น้ำ"


def test_quality_gate_flags_corruption():
    assert thai_quality_issues("คุ้มครองชีวิต") == 0
    assert thai_quality_issues("ต้องแนบสัญญาเพิ่มเติมการประกััุัิุ") > 0   # stacked vowels from a bad table split
    assert thai_quality_issues("| ํิัััิ่ |") > 0                       # orphan marks


@pytest.mark.parametrize("text,expected", [
    ("50,000", 50_000), ("5 หมื่น", 50_000), ("3.5 หมื่น", 35_000), ("35k", 35_000), ("1.2 แสน", 120_000),
    ("สามหมื่นห้า", 35_000), ("หนึ่งแสนสองหมื่น", 120_000), ("3 หมื่น 5", 35_000), ("๔๕,๐๐๐ บาท", 45_000),
    ("ไม่บอก", None),
])
def test_parse_thai_amount(text, expected):
    assert parse_thai_amount(text) == expected


def test_tokenize_keeps_product_codes_distinct():
    a, b = tokenize("ตลอดชีพ 90/10"), tokenize("ตลอดชีพ 90/20")
    assert "code90x10" in a and "code90x20" in b and "code90x10" not in b


def test_char_ngrams_segmentation_free():
    assert "ประ" in char_ngrams("เบี้ยประกันภัย")
    assert char_ngrams("1,000,000")[0] == "100"


def test_detect_products_and_synonyms():
    assert detect_products("เทียบ 90/20 กับ ตลอดชีพ 99/99") == ["whole_life_90_20", "whole_life_99_99"]
    assert detect_products("คุ้มชีวา กับ คุ้มชีวี ต่างกันยังไง") == ["khum_cheeva_90_10", "khum_cheevee_extra_99"]
    assert detect_products("190/100") == []
    assert "จำนวนเงินเอาประกันภัย" in expand_query("ทุนประกัน 5 แสน")


# --------------------------------------------------------------------------- parser
@pytest.fixture(scope="module")
def parsed():
    return {p.product_id: parse_pdf(PDF_DIR / p.file_name) for p in PRODUCTS}


def test_all_pages_extract_without_thai_corruption(parsed):
    for pid, doc in parsed.items():
        assert doc.pages, pid
        for page in doc.pages:
            assert page.quality_issues == 0, (pid, page.page_number)
            assert "" not in page.markdown and "" not in page.markdown


def test_tables_keep_label_value_pairs(parsed):
    md = parsed["khum_cheeva_90_10"].markdown
    assert "| เบี้ยประกันภัยรวมรายปี (ปีแรก) เพศชาย อายุ 40 ปี ขั้นอาชีพ 1 | 18,828 | 30,880 | 61,760 |" in md
    # premium table: male column must stay before female (a generic converter swapped them)
    assert "| 40 | 5,775 | 4,830 |" in parsed["whole_life_99_99"].markdown
    assert "| อายุรับประกันภัย | 20 - 60 ปี | 20 - 55 ปี | 20 - 50 ปี |" in parsed["khum_manjai_sure_term"].markdown


def test_feature_boxes_not_interleaved(parsed):
    md = parsed["whole_life_90_20"].markdown
    # each 4-column feature box must come out as one contiguous block
    assert "### คุ้มภาษี\n\nลดหย่อนภาษีสูงสุด\n100,000 บาท\nตามเงื่อนไขกรมสรรพากร" in md


# --------------------------------------------------------------------------- chunker
def test_chunks_have_context_and_metadata(parsed):
    parents, children = build_chunks(parsed["whole_life_90_20"])
    assert {c.metadata["product_id"] for c in children} == {"whole_life_90_20"}
    row = next(c for c in children if "เพศหญิง อายุ 35 ปี" in c.text and c.metadata["chunk_type"] == "table_row")
    assert row.text.startswith("ตลอดชีพ 90/20 > ตัวอย่างเบี้ยประกันภัยรายปี")
    assert "แผน 3 (ทุนประกัน 500,000 บาท) = 10,650" in row.text       # plan -> sum assured made explicit
    assert {c.parent_id for c in children} <= {p.parent_id for p in parents}


def test_column_headers_do_not_leak_across_sections(parsed):
    _, children = build_chunks(parsed["whole_life_99_99"])
    uw = [c.text for c in children if c.metadata["section"] == "underwriting" and c.metadata["chunk_type"] == "table_row"]
    assert uw and not any("หญิง =" in t for t in uw)
