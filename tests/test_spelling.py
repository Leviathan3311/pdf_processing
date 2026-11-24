"""Pure-function tests for the constrained diacritic correction (no model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conversion.spelling import _slots, accept, is_locked, merge, skeleton, split_sentences  # noqa: E402


def test_skeleton_strips_accents_and_d():
    assert skeleton("Trưởng đoàn") == "Truong doan"
    assert skeleton("ĐỀ XUẤT") == "DE XUAT"


def test_accept_only_diacritic_changes():
    assert accept("tối", "tội")
    assert accept("dón", "đón")
    assert not accept("sói", "thời")          # different letters
    assert not accept("Tham", "tham")         # case change
    assert not accept("tội", "tội")           # no change


def test_locked_tokens_never_change():
    for token in ["A05", "12h30", "B5D13", "K7.HND(05b).", "11552/TB-A05-TT2", "k/gùi;", "CN&ATTT", "ngày04"]:
        assert is_locked(token)
        assert not accept(token, token + "x")
    assert not is_locked("Lưu:")


def test_merge_keeps_whitespace_and_rejects_bad_changes():
    original = "HỌC VIỆN\nVÀ tối phạm  Luu: A05"
    rewrite = "HỌC VIỆN VÀ tội phạm Lưu: A06"
    fixed, changes = merge(original, rewrite)
    assert fixed == "HỌC VIỆN\nVÀ tội phạm  Lưu: A05"
    assert changes == [("tối", "tội"), ("Luu:", "Lưu:")]


def test_merge_ignores_inserted_or_dropped_words():
    fixed, changes = merge("đề đưa đón", "để đưa và đón")
    assert fixed == "để đưa đón"
    assert changes == [("đề", "để")]


def test_split_sentences_limits_length():
    text = "Câu một dài. " * 40
    chunks = split_sentences(text, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks) == " ".join(text.split())


def test_slots_never_include_latex():
    middle = {"pages": [{"page_idx": 0, "blocks": [
        {"type": "text", "index": 0, "content": [
            {"type": "text", "content": "a) tối phạm "},
            {"type": "equation_inline", "content": "x^{2} - 4x + 3 \\leq 0"}]},
        {"type": "interline_equation", "index": 1, "content": "\\frac{2x}{1+x^2}"},
    ]}]}
    assert [s.text for s in _slots(middle)] == ["a) tối phạm "]


def test_slots_cover_text_and_cells_and_skip_images():
    middle = {"pages": [{"page_idx": 0, "blocks": [
        {"type": "text", "index": 0, "content": [{"type": "text", "content": "tối phạm"}]},
        {"type": "image", "index": 1, "content": [{"type": "image_body", "content": "99"}]},
        {"type": "table", "index": 2, "content": [
            {"type": "table_body", "content": "<table><tr><td>Trường đoàn</td><td rowspan=\"2\">A05</td></tr></table>"},
            {"type": "table_footnote", "content": [{"type": "text", "content": "Ấn định"}]}]},
    ]}]}
    slots = _slots(middle)
    assert [s.text for s in slots] == ["tối phạm", "Trường đoàn", "A05", "Ấn định"]
    slots[1].write("Trưởng đoàn")
    html = middle["pages"][0]["blocks"][2]["content"][0]["content"]
    assert html == "<table><tr><td>Trưởng đoàn</td><td rowspan=\"2\">A05</td></tr></table>"
