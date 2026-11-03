import pytest


@pytest.mark.parametrize("question,product,section,fact", [
    ("คุ้มชีวา 90/10 เบี้ยรายปี ผู้ชาย 40 แผน 1", "khum_cheeva_90_10", "benefit_example", "18,828"),
    ("ตลอดชีพ 90/20 ผู้หญิงอายุ 35 ทุน 500,000 เบี้ยปีละเท่าไหร่", "whole_life_90_20", "premium_example", "10,650"),
    ("อายุ 72 สมัครตลอดชีพ 99/99 ต้องแนบอะไร", "whole_life_99_99", "underwriting", "คุ้มรักษาเหมาจ่าย"),
    ("ประกันชั่วระยะเวลาแบบ 15 ปี รับคนอายุเท่าไหร่", "khum_manjai_sure_term", "underwriting", "20 - 50 ปี"),
])
def test_top_parent_is_answer_bearing(retriever, question, product, section, fact):
    res = retriever.retrieve(question)
    top = res.parents[0]
    assert top.metadata["product_id"] == product
    assert any(fact in p.text for p in res.parents[:2])


def test_product_filter_excludes_other_products(retriever):
    res = retriever.retrieve("เบี้ยประกันเท่าไหร่", products=["whole_life_90_20"], use_filter=True)
    assert res.parents and {p.metadata["product_id"] for p in res.parents} == {"whole_life_90_20"}


def test_comparison_represents_every_named_product(retriever):
    res = retriever.retrieve("เปรียบเทียบ 90/20 กับ 99/99 จ่ายเบี้ยกี่ปี")
    assert {"whole_life_90_20", "whole_life_99_99"} <= {p.metadata["product_id"] for p in res.parents}


def test_citation_format(retriever):
    p = retriever.retrieve("คุ้มชีวา 90/10 อายุรับประกันภัย").parents[0]
    assert "khum_cheeva_90_10.pdf" in p.citation() and "หน้า" in p.citation()
