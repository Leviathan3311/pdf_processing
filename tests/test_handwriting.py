"""Checks on the VLM answer for handwritten blocks (docs/handwriting-ocr/DESIGN.md section 5)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conversion.handwriting import Region, _prose_block, _red_ink, _spans, apply, check  # noqa: E402


def test_red_seal_ink_is_not_sent_but_blue_pen_is():
    assert _red_ink("C0393F")          # seal text
    assert not _red_ink("2A3F9A")      # blue ballpoint note
    assert not _red_ink(None)          # black print


def test_check_accepts_plain_vietnamese_reading():
    assert check("Vì sao phải scale dữ liệu", "stop", 8.0, reference="Vi sao phai scale du lieu") == ""


def test_check_rejects_empty_truncated_and_explanations():
    assert check("<EMPTY>", "stop", 8.0) == "empty"
    assert check("abc def", "length", 8.0) == "max_tokens"
    assert check("Đây là nội dung trong ảnh: abc", "stop", 8.0) == "explanation"


def test_check_length_is_relative_to_mineru_reading():
    assert check("ab", "stop", 8.0, reference="một câu khá dài của MinerU").startswith("length")
    long = "chữ " * 40
    assert check(long, "stop", 8.0, reference="ngắn thôi").startswith("length")


def test_check_rejects_prose_spelled_as_math_and_raw_latex():
    assert check(r"$$N \in \theta s c a l e$$", "stop", 4.0, reference="N thuộc theta scale") == "word as math"
    assert check(r"x \frac{1}{2} y", "stop", 4.0, reference="x một phần hai y") == "latex outside $"


def test_check_keeps_short_math_variables():
    assert check(r"Với $x_1$ và $y$ ta có", "stop", 6.0, reference="Với x1 và y ta có") == ""


def test_check_rejects_repeats_and_foreign_script():
    assert check("a b c " * 6, "stop", 20.0, reference="a b c " * 6) == "repeats"
    assert check("[mɛ̃] [ɑ̃] [ɔ̃]", "stop", 4.0, reference="[mẽ] [ã] [õ]") == "script"


def test_spans_split_prose_and_math_and_mark_hw():
    spans = _spans(r"Ta có $x^2$ nên")
    assert [s["type"] for s in spans] == ["text", "equation_inline", "text"]
    assert spans[1]["content"] == "x^2"
    assert all(s["hw"] and s["src"] == "vlm" for s in spans)


def test_spans_turn_bare_subscripts_into_math_but_not_snake_case():
    spans = _spans("Output = X_{proj} + Z, file_name")
    assert [(s["type"], s["content"]) for s in spans] == [
        ("text", "Output = "), ("equation_inline", "X_{proj}"), ("text", " + Z, file_name")]


def test_prose_block_skips_tall_math_and_pictures():
    assert _prose_block({"type": "text", "bbox": [0, 0, 1, 1], "content": "Mô hình học máy"})
    assert not _prose_block({"type": "image", "bbox": [0, 0, 1, 1], "content": "x"})
    frac = [{"type": "equation_inline", "content": r"\frac{a+b}{c}"}]
    assert not _prose_block({"type": "text", "bbox": [0, 0, 1, 1], "content": frac})


def _region(block, text, mineru="Vi sao phai scale"):
    return Region("u", block, "T2", b"", expected_chars=20, line_units=6.0, mineru_text=mineru, colored=False,
                  text=text, finish="stop")


def test_apply_writes_vlm_text_and_retypes_equation_blocks():
    block = {"type": "interline_equation", "content": r"\text{Vi sao phai scale}", "bbox": [0, 0, 1, 1]}
    r = _region(block, "Vì sao phải scale")
    apply(r)
    assert r.verdict == "accepted"
    assert block["type"] == "text" and block["hw"]
    assert block["content"][0]["content"] == "Vì sao phải scale"


def test_apply_keeps_mineru_on_failed_check_or_signature():
    block = {"type": "text", "content": "Vi sao phai scale", "bbox": [0, 0, 1, 1]}
    r = _region(block, "<EMPTY>")
    apply(r)
    assert r.verdict == "fallback:empty" and block["content"] == "Vi sao phai scale"
    r = _region(block, "<SIGNATURE>")
    apply(r)
    assert r.verdict == "signature" and "hw" not in block


def test_apply_flags_review_when_readings_disagree():
    block = {"type": "text", "content": "abc", "bbox": [0, 0, 1, 1]}
    r = _region(block, "Hoàn toàn khác nhau", mineru="Mot cau khac")
    apply(r)
    assert r.verdict == "accepted" and r.review and block["needs_review"]


def test_apply_keeps_picture_for_disputed_note_on_printed_page():
    block = {"type": "text", "content": "Amy", "bbox": [0, 0, 1, 1]}
    r = _region(block, "Đặng", mineru="Amy")
    r.signal = "T1"
    apply(r)
    assert r.verdict == "fallback:disagree" and "hw" not in block
