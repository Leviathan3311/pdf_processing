"""PDF input: per-page classification and direct text-layer extraction (PyMuPDF only, no models, no Word)."""
import sys
from pathlib import Path

import fitz
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conversion.layout import Para, TableItem  # noqa: E402
from conversion.page_classifier import PageKind, classify_page  # noqa: E402
from conversion.pymupdf_extractor import extract_page  # noqa: E402

FONT = "C:/Windows/Fonts/times.ttf"
BOLD = "C:/Windows/Fonts/timesbd.ttf"


def _picture_page(doc, hidden_text: str | None = None):
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 420), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    page.insert_image(page.rect, pixmap=pix)
    if hidden_text:                                    # a scanner's invisible OCR layer
        page.insert_text((72, 100), hidden_text, render_mode=3)
    return page


def test_classifier_routes_each_page():
    doc = fitz.open()
    text = doc.new_page(width=595, height=842)
    text.insert_font(fontname="tnr", fontfile=FONT)
    text.insert_text((72, 100), "Cộng hoà xã hội chủ nghĩa Việt Nam", fontname="tnr", fontsize=13)
    _picture_page(doc)
    _picture_page(doc, hidden_text="UY NUAN DAN THANH Ho Ho cui tvtrNu " * 3)
    maths = doc.new_page(width=595, height=842)
    maths.insert_font(fontname="tnr", fontfile=FONT)
    maths.insert_text((72, 100), "x ≤ y, a ≠ b, ∑ 1/n = ∞, ∂f", fontname="tnr", fontsize=13)
    kinds = [classify_page(p)[0] for p in doc]
    assert kinds == [PageKind.TEXT_SIMPLE, PageKind.SCAN, PageKind.SCAN, PageKind.TEXT_COMPLEX]


def test_text_layer_extraction_keeps_fonts_lines_and_tables(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="tnr", fontfile=FONT)
    page.insert_font(fontname="tnrb", fontfile=BOLD)
    page.insert_text((200, 80), "KẾ HOẠCH", fontname="tnrb", fontsize=14)
    page.insert_text((72, 120), "Số: 12/KH-SGD", fontname="tnr", fontsize=13, color=(0.8, 0, 0))
    # a 2 x 3 ruled table whose last column is merged over both rows
    xs, ys = [72, 200, 330, 460], [200, 230, 260]
    for y in (ys[0], ys[-1]):
        page.draw_line((xs[0], y), (xs[-1], y))
    page.draw_line((xs[0], ys[1]), (xs[2], ys[1]))      # the middle rule stops before the merged column
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for (x, y), t in zip([(80, 220), (210, 220), (80, 250), (210, 250), (340, 235)],
                         ["STT", "Môn", "1", "Toán", "Ghi chú"]):
        page.insert_text((x, y), t, fontname="tnr", fontsize=12)

    built = extract_page(page, tmp_path)
    paras = [it for it in built.flow if isinstance(it, Para)]
    texts = {"".join(r.text for r in p.lines[0]).replace(" ", " ").replace("­", "-"): p.lines[0][0].typo for p in paras}
    assert texts["KẾ HOẠCH"].bold and texts["KẾ HOẠCH"].size == 14.0
    assert texts["Số: 12/KH-SGD"].color == "CC0000" and not texts["Số: 12/KH-SGD"].bold
    tables = [it for it in built.flow if isinstance(it, TableItem)]
    assert len(tables) == 1
    assert 'rowspan="2"' in tables[0].html and "Ghi chú" in tables[0].html
    assert np.allclose(tables[0].col_edges, xs, atol=1.5)
    assert "STT" not in texts                         # table text stays in the table
