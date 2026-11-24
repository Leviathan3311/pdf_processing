"""Per-page PDF classification (CLAUDE.md section 4): which extractor each page needs.

    SCAN          no usable text layer (a picture of a page, a hidden OCR layer, garbled or outlined text)
                  → render the page to an image and OCR it (MinerU standard)
    TEXT_COMPLEX  a real text layer with formulas → render + MinerU (formulas come back as LaTeX → Word equations)
    TEXT_SIMPLE   a real text layer → read text, fonts, tables and images straight from the PDF (exact, ~ms)

Only cheap PyMuPDF signals; thresholds are starting values (tune on samples).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import fitz  # PyMuPDF
import numpy as np


class PageKind(str, Enum):
    SCAN = "scan"
    TEXT_COMPLEX = "text_complex"
    TEXT_SIMPLE = "text_simple"


MATH_FONT = re.compile(r"CMMI|CMSY|CMEX|CMBSY|MSBM|Cambria ?Math|CambriaMath|Symbol|STIX|Math", re.I)


def _is_math_char(ch: str) -> bool:
    o = ord(ch)
    return 0x2200 <= o <= 0x22FF or 0x1D400 <= o <= 0x1D7FF or 0x27C0 <= o <= 0x27EF or 0x2980 <= o <= 0x2AFF


def _is_garbled(ch: str) -> bool:
    o = ord(ch)
    return ch == "�" or 0xE000 <= o <= 0xF8FF or (o < 32 and ch not in "\t\n\r")


@dataclass
class PageSignals:
    visible_chars: str           # text of visible spans (hidden OCR layers removed)
    hidden_spans: int            # spans drawn invisibly (render mode 3): a scanner's OCR layer
    image_coverage: float        # 0..1, union of image boxes / page area
    garbled: float               # 0..1 share of U+FFFD / private-use / control characters
    drawings: int
    math: bool


def signals(page: fitz.Page) -> PageSignals:
    visible, hidden, math_font = [], 0, False
    for span in page.get_texttrace():
        if span["type"] == 3:                       # invisible text
            hidden += 1
            continue
        visible.append("".join(chr(c[0]) for c in span["chars"]))
        math_font = math_font or bool(MATH_FONT.search(span.get("font", "")))
    text = "".join(visible)
    chars = [c for c in text if not c.isspace()]

    w, h = int(page.rect.width) + 1, int(page.rect.height) + 1
    mask = np.zeros((h, w), bool)
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        if not r.is_empty:
            mask[int(r.y0):int(r.y1) + 1, int(r.x0):int(r.x1) + 1] = True
    coverage = float(mask.mean())

    garbled = sum(map(_is_garbled, chars)) / len(chars) if chars else 0.0
    math_chars = sum(map(_is_math_char, chars))
    return PageSignals("".join(chars), hidden, coverage, garbled, len(page.get_drawings()),
                       math_font or math_chars >= 3)


def classify(sig: PageSignals) -> tuple[PageKind, str]:
    """Decision order of CLAUDE.md 4.2. Returns the kind and the reason (logged)."""
    n = len(sig.visible_chars)
    if n == 0 and sig.image_coverage > 0.3:
        return PageKind.SCAN, "no text over an image"
    if sig.image_coverage > 0.85 and n < 50:
        return PageKind.SCAN, "full-page image"
    if sig.hidden_spans and sig.image_coverage > 0.85:
        return PageKind.SCAN, "scanner OCR layer (ignored, re-OCR)"
    if n and sig.garbled > 0.2:
        return PageKind.SCAN, f"garbled text layer ({sig.garbled:.0%})"
    if n == 0 and sig.drawings > 500:
        return PageKind.SCAN, "text drawn as outlines"
    if sig.math:
        return PageKind.TEXT_COMPLEX, "formulas"
    return PageKind.TEXT_SIMPLE, "text layer"


def classify_page(page: fitz.Page) -> tuple[PageKind, str]:
    return classify(signals(page))
