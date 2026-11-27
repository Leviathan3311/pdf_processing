"""Tests for layout measurement, XY-cut and the DOCX renderer (no OCR model, no Word needed)."""
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conversion.docx_renderer import render_pages  # noqa: E402
from conversion.layout import (Page, Para, Run, TableItem, VStack, HSplit, _cells, _grid_shape,  # noqa: E402
                               page_size, xy_cut)
from conversion.measure import (Typo, _font, cluster_sizes, fit_single_line, ink_mask, split_by_widths,  # noqa: E402
                                text_lines, text_width_pt, wrap)


def _render_lines(lines, size_px=40, width=1400):
    """White image with each text line drawn in Times New Roman, like a clean scan."""
    font = _font("Times New Roman", False, False).font_variant(size=size_px)
    img = Image.new("L", (width, int(size_px * 1.6) * len(lines) + 40), 255)
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((40, 20 + i * int(size_px * 1.6)), text, font=font, fill=0)
    return np.array(img)


def test_text_lines_counts_vietnamese_lines_despite_diacritics():
    gray = _render_lines(["Cộng hòa xã hội chủ nghĩa Việt Nam", "Độc lập – Tự do – Hạnh phúc",
                          "Ấn định danh sách gồm 10 thành viên"])
    assert len(text_lines(ink_mask(gray))) == 3


def test_text_lines_follow_long_slanted_lines():
    text = "những dòng chữ viết tay dài trên tờ giấy bị nghiêng lên"
    gray = _render_lines([text] * 5, size_px=36, width=1300)
    h, w = gray.shape
    rot = cv2.warpAffine(gray, cv2.getRotationMatrix2D((w / 2, h / 2), 4, 1.0), (w, h + 120),
                         borderValue=255)                  # lines drift 4° across the block
    assert len(text_lines(ink_mask(rot))) == 5


def test_text_lines_ignore_a_bent_hand_drawn_column_divider():
    gray = _render_lines(["dòng một của cột", "dòng hai của cột", "dòng ba của cột", "dòng bốn của cột",
                          "dòng năm của cột"], width=900)
    h = gray.shape[0]
    pts = np.array([[860 + int(6 * np.sin(y / 40)), y] for y in range(10, h - 10)], np.int32)
    cv2.polylines(gray, [pts], False, 0, 4)                    # wavy vertical pen line beside the text
    assert len(text_lines(ink_mask(gray))) == 5


def test_fit_single_line_recovers_size():
    text = "Trung tá, Nguyễn Văn An"
    assert abs(fit_single_line(text, text_width_pt(text, 13.0), False, False) - 13.0) < 1e-6


def test_wrap_respects_first_line_indent():
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    plain = wrap(text, 120, 12, "Times New Roman", False, False)
    indented = wrap(text, 120, 12, "Times New Roman", False, False, first_indent=40)
    assert len(indented[0].split()) < len(plain[0].split())


def test_split_by_widths_keeps_words_and_order():
    text = "ĐỀ XUẤT V/v Tham dự cuộc thi Hành trình truy vết tội phạm mạng"
    parts = split_by_widths(text, [70, 400])
    assert parts == ["ĐỀ XUẤT", "V/v Tham dự cuộc thi Hành trình truy vết tội phạm mạng"]


def test_cluster_sizes_pulls_noise_together():
    out = cluster_sizes([13.1, 13.6, 12.9, 18.0])
    assert out[0] == out[1] == out[2] and out[3] == 18.0


def test_page_size_detects_a4_portrait_and_landscape():
    assert page_size(1712, 2400) == (595.3, 841.9)
    assert page_size(2400, 1679) == (841.9, 595.3)


def test_grid_shape_and_cells_follow_spans():
    html = ('<table><tr><td>STT</td><td>Ghi chú</td></tr>'
            '<tr><td>1</td><td rowspan="2">Tham gia</td></tr><tr><td>2</td></tr></table>')
    assert _grid_shape(html) == (3, 2)
    assert (1, 1, 2, 1, "Tham gia") in _cells(html)
    assert (2, 0, 1, 1, "2") in _cells(html)


def test_tokens_keep_formulas_separate_and_lines():
    from conversion.layout import _plain, _runs, _segments, _token_lines, _tokens
    content = [{"type": "text", "content": "a) "}, {"type": "equation_inline", "content": "x^{2} - 4x + 3 = 0"},
               {"type": "text", "content": ";\nb) "}, {"type": "equation_inline", "content": "\\frac{1}{2}"}]
    lines = _token_lines(_tokens(_segments(content)))
    assert len(lines) == 2
    runs = _runs(lines[0], Typo(size=12))
    assert [r.latex for r in runs] == [None, "x^{2} - 4x + 3 = 0", None]
    assert runs[0].text == "a) " and runs[2].text == ";"
    assert "4x" in _plain(lines[0]) and "\\" not in _plain(lines[1])      # widths use shown text, not LaTeX


def test_match_lines_wrap_then_grid():
    from conversion.layout import _match_lines
    # MinerU lines (unit widths at 1 pt): a long sentence, then a) b) c) d) short formulas
    units = [60.0, 8.0, 8.0, 8.0, 8.0]
    # scan: the sentence wraps over two 420 pt lines, then two grid rows "a) … b) …" with a 150 pt gap
    widths = [420.0, 300.0, 250.0, 250.0]
    gaps = [0.0, 0.0, 150.0, 150.0]
    plan = _match_lines(units, widths, gaps)
    assert plan == [("span", 0, 0, 1), ("group", 1, 2, 2), ("group", 3, 4, 3)]


def test_match_lines_one_to_one():
    from conversion.layout import _match_lines
    assert _match_lines([10.0, 14.0], [130.0, 180.0], [0.0, 0.0]) == [("span", 0, 0, 0), ("span", 1, 1, 1)]


def test_fraction_width_follows_structure():
    from conversion.math_omml import width_em
    assert width_em("\\frac{2x}{1+x^2}") < 0.7 * width_em("2x + 1 + x^2")
    assert width_em("x^{2}") < width_em("x2")


def test_latex_becomes_native_word_equation(tmp_path):
    from conversion.math_omml import latex_to_omml
    omml = latex_to_omml("\\frac{2x}{1+x^2}", 13)
    assert omml is not None and omml.tag.endswith("}oMath")
    flow = [Para(80, 550, [[Run("f(x) = ", Typo(size=13)), Run("2x/(1+x2)", Typo(size=13), latex="\\frac{2x}{1+x^2}")]],
                 [100], 16.0)]
    xml = zipfile.ZipFile(render_pages([Page(595.3, 841.9, flow)], tmp_path / "m.docx")).read("word/document.xml")
    assert b"<m:oMath" in xml and b"<m:f>" in xml and b"\\frac" not in xml


def _para(x0, x1, y, text="x"):
    return Para(x0, x1, [[Run(text, Typo(size=12))]], [y], 14.0)


def test_xy_cut_keeps_matrices_side_by_side_under_a_tall_formula():
    def para(x0, x1, y0, y1):
        return Para(x0, x1, [[]], [(y0 + y1) / 2], y1 - y0)
    formula = para(40, 420, 116, 201)                 # its line height reaches into the next row
    wq, wk, wv = para(44, 134, 187, 276), para(173, 261, 187, 267), para(298, 382, 181, 261)
    node = xy_cut([formula, wq, wk, wv], 0, 595)
    assert isinstance(node, VStack) and node.children[0] is formula
    split = node.children[1]
    assert isinstance(split, HSplit) and len(split.columns) == 3


def test_xy_cut_two_column_letterhead_then_body():
    left, right = _para(70, 240, 40, "HỌC VIỆN"), _para(280, 560, 40, "CỘNG HÒA")
    body = _para(80, 550, 200, "Căn cứ")
    root = xy_cut([left, right, body], 60, 560)
    assert isinstance(root, VStack)
    assert isinstance(root.children[0], HSplit) and root.children[1] is body
    cols = root.children[0].columns
    assert cols[0].children == [left] and cols[1].children == [right]


def test_rendered_docx_is_editable_flow(tmp_path):
    flow = [_para(70, 240, 40, "HỌC VIỆN KỸ THUẬT"), _para(280, 560, 40, "CỘNG HÒA XÃ HỘI"),
            Para(80, 550, [[Run("Căn cứ văn bản số 11552 về lịch tổ chức vòng chung kết.", Typo(size=13))]],
                 [120, 138], 18.0, wrap=True, justify=True, first_indent=28),
            TableItem(80, 160, 550, 220, '<table><tr><td>STT</td><td>Ghi chú</td></tr>'
                      '<tr><td>1</td><td>Tham gia</td></tr></table>', size=12.0)]
    out = render_pages([Page(595.3, 841.9, flow)], tmp_path / "t.docx")
    xml = zipfile.ZipFile(out).read("word/document.xml").decode("utf-8")
    assert "txbxContent" not in xml                    # text never in text boxes
    assert "Căn cứ văn bản" in xml and "Tham gia" in xml
    assert xml.count("<w:tbl>") == 2                   # letterhead split + real table


MATRIX = r"A = \left[ \begin{array}{l l} 2 & 0 \\ 0 & 2 \\ 1 & 0 \end{array} \right]"


def test_matrix_is_tall_and_measured_per_column():
    from conversion.math_omml import height_em, width_em
    assert height_em(MATRIX) > 3.0 and height_em("e^{x} + 1") < 1.5
    assert height_em(r"\hat{a} + \underline{vd}") < 1.5         # accents are not stacked rows
    assert width_em(MATRIX) < width_em("A = [2 0 0 2 1 0]") + 2.0   # widest cell per column, not every cell


def test_matrix_brackets_become_a_stretching_word_delimiter():
    from lxml import etree
    from conversion.math_omml import latex_to_omml
    xml = etree.tostring(latex_to_omml(MATRIX, 12)).decode()
    assert "<m:d>" in xml or "<m:d " in xml                       # m:d grows with the 3 rows
    assert "<m:m>" in xml or "<m:m " in xml


def test_ink_reference_ignores_background_outside_text():
    from conversion.measure import ink_reference, is_pen_page
    img = np.full((400, 400, 3), 255, np.uint8)
    img[:, :60] = (120, 80, 40)                                    # brown table top beside the sheet
    for y in range(120, 300, 30):                                  # blue pen lines on the sheet
        img[y:y + 6, 100:360] = (40, 50, 140)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ref = ink_reference(img, gray, [(90, 110, 370, 310)])
    assert ref[2] > 0 > ref[0]                                     # blue, not the brown background
    assert is_pen_page(ref)
    black = np.full_like(img, 255)
    black[150:160, 100:360] = 20
    assert not is_pen_page(ink_reference(black, cv2.cvtColor(black, cv2.COLOR_RGB2GRAY), [(90, 140, 370, 170)]))


def test_paragraph_after_a_nested_split_stays_below_it(tmp_path):
    # column 1 of a band = [two-column sub-band, then a line]: the line must come after the sub-band
    top_l, top_r = _para(70, 140, 40, "Ta dang co"), _para(150, 300, 40, "Score")
    below = _para(70, 230, 90, "Chia tung phan tu")
    side = Para(330, 560, [[Run("Scale", Typo(size=12))], [Run("why", Typo(size=12))]], [40, 90], 14.0)
    out = render_pages([Page(595.3, 841.9, [top_l, top_r, below, side])], tmp_path / "n.docx")
    xml = zipfile.ZipFile(out).read("word/document.xml").decode("utf-8")
    assert xml.index("Score") < xml.index("Chia tung phan tu")
