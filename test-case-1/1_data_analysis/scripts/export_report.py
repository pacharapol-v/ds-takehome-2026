"""Export outputs/report.html to static copies: report.pdf (A4) and report.svg (the whole page as one image).

Run from test-case-1/1_data_analysis, after build_report.py:
    python scripts/export_report.py
Needs: pip install playwright pymupdf, and Google Chrome installed (used through playwright's "chrome" channel,
so no separate browser download is needed).
"""
from pathlib import Path

import pymupdf
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "outputs" / "report.html"
PDF = ROOT / "outputs" / "report.pdf"
SVG = ROOT / "outputs" / "report.svg"

# A4 minus 10 mm margins is 190 mm = 718 CSS px. Printing at 75% lays the page out 957 px wide,
# so the viewport is set to the same width and the charts are drawn for the width they print at.
PDF_SCALE = 0.75
PDF_WIDTH = round(718 / PDF_SCALE)
SVG_WIDTH = 1280  # desktop layout, as seen on screen


def load(browser, width, media):
    page = browser.new_page(viewport={"width": width, "height": 900})
    page.emulate_media(media=media)
    page.goto(HTML.as_uri(), wait_until="load")
    page.wait_for_function("document.querySelectorAll('.js-plotly-plot .main-svg').length > 0")
    page.wait_for_timeout(500)  # let Plotly finish its responsive resize
    return page


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")

    page = load(browser, PDF_WIDTH, "print")
    page.pdf(path=str(PDF), format="A4", scale=PDF_SCALE, print_background=True,
             margin={"top": "10mm", "bottom": "10mm", "left": "10mm", "right": "10mm"})

    # SVG: print the screen layout onto one page exactly as tall as the content, then convert that page.
    page = load(browser, SVG_WIDTH, "screen")
    height = page.evaluate("document.documentElement.scrollHeight")
    one_page = page.pdf(width=f"{SVG_WIDTH}px", height=f"{height + 2}px", print_background=True,
                        margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    browser.close()

doc = pymupdf.open(stream=one_page, filetype="pdf")
SVG.write_text(doc[0].get_svg_image(), encoding="utf-8")  # text drawn as paths, so it looks the same without the fonts

print(f"wrote {PDF} ({pymupdf.open(PDF).page_count} pages, {PDF.stat().st_size / 1e6:.1f} MB)")
print(f"wrote {SVG} ({SVG.stat().st_size / 1e6:.1f} MB, {doc.page_count} page from the screen layout)")
