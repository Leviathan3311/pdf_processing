"""Closed-loop position check (CLAUDE.md section 3, step 6).

After Word has laid out the DOCX, read where each paragraph's first line really landed in the exported PDF
and move that paragraph by the difference. A paragraph's spacing is computed from model coordinates, so
shifting one paragraph's line centres moves only that paragraph (the next one is measured from the model).
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

from .layout import Page, Para
from .measure import _font, _strip_marks


def _key(text: str) -> str:
    return "".join(unicodedata.normalize("NFC", text).split())[:14]


def pdf_lines(pdf: Path) -> list[list[tuple[str, float]]]:
    """Per page: (text, baseline y in pt) of every text line Word produced."""
    out = []
    for page in fitz.open(str(pdf)):
        lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text:
                    lines.append((text, line["spans"][0]["origin"][1]))
        out.append(lines)
    return out


def ink_centre_offset(text: str, typo) -> float:
    """Distance (pt, negative = above) from the baseline to the centre of the base-letter ink — the same
    quantity the scan measurement uses (diacritics are separate components there)."""
    left, top, right, bottom = _font(typo.font, typo.bold, typo.italic).getbbox(_strip_marks(text) or "x",
                                                                                 anchor="ls")
    return (top + bottom) / 2 / 1000.0 * typo.size


OVERFLOW_PENALTY = 100.0       # a render that spills onto an extra page is worse than any position error


def snapshot_positions(pages: list[Page]) -> list[list[tuple]]:
    return [[(list(it.centres), it.line_height) if isinstance(it, Para) else None for it in p.flow] for p in pages]


def restore_positions(pages: list[Page], snap: list[list[tuple]]) -> None:
    for page, saved in zip(pages, snap):
        for item, state in zip(page.flow, saved):
            if isinstance(item, Para) and state is not None:
                item.centres, item.line_height = list(state[0]), state[1]


def shrink_lines(pages: list[Page], factor: float = 0.96) -> None:
    """Word spilled onto an extra page (tables and exact line heights add a little height each):
    tighten every line a few percent, never below the font size, and render again."""
    for page in pages:
        for item in page.flow:
            if isinstance(item, Para):
                size = max(r.typo.size for line in item.lines for r in line)
                item.line_height = max(size, item.line_height * factor)


def correct_positions(pages: list[Page], pdf: Path, max_shift: float = 12.0) -> float:
    """Shift each paragraph by (target − rendered) of its first line. Returns the mean |error| before."""
    rendered = pdf_lines(pdf)
    errors = []
    for page, lines in zip(pages, rendered):
        used: set[int] = set()
        for item in page.flow:
            if not isinstance(item, Para) or not item.lines:
                continue
            key = _key(item.lines[0][0].text)
            match = next((i for i, (t, _) in enumerate(lines) if i not in used and _key(t) == key), None)
            if match is None:
                continue
            used.add(match)
            if item.target is None:
                item.target = item.centres[0]
            run = item.lines[0][0]
            rendered = lines[match][1] + ink_centre_offset(lines[match][0], run.typo)
            delta = item.target - rendered                  # + : Word put it too high, − : too low
            errors.append(abs(delta))
            if abs(delta) <= max_shift:
                item.centres = [c + delta for c in item.centres]
    return sum(errors) / len(errors) if errors else 0.0
