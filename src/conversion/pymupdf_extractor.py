"""TEXT_SIMPLE pages: read the layout straight from the PDF text layer (CLAUDE.md step 2a).

Nothing is measured or guessed: text, font, size, bold / italic, colour and baseline come from the PDF;
ruled tables from `find_tables()` (merged cells from the cell boxes); pictures from the image list; thin
vector rules (letterhead underlines) are redrawn. The result is the same `layout.Page` the OCR path builds,
so the renderer and the closed loop treat both alike.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from .fit_check import ink_centre_offset
from .layout import FloatImage, Page, Para, Run, TableItem
from .measure import Typo, snap_size, text_width_pt

SANS = ("arial", "helvetica", "calibri", "verdana", "tahoma", "segoe", "sans")


def _typo(span: dict) -> Typo:
    name = span["font"].lower()
    flags = span["flags"]
    font = "Arial" if any(s in name for s in SANS) and "serif" not in name.replace("sans", "") else "Times New Roman"
    bold = bool(flags & 16) or any(s in name for s in ("bold", "black", "heavy", "semibold"))
    italic = bool(flags & 2) or "italic" in name or "oblique" in name
    c = span["color"]
    r, g, b = (c >> 16) & 255, (c >> 8) & 255, c & 255
    color = None if max(r, g, b) < 60 else f"{r:02X}{g:02X}{b:02X}"
    return Typo(size=snap_size(span["size"]), font=font, bold=bold, italic=italic, color=color)


@dataclass
class _Line:
    runs: list[Run]
    x0: float
    x1: float
    baseline: float
    size: float

    @property
    def text(self) -> str:
        return "".join(r.text for r in self.runs)


def _runs(spans: list[dict]) -> list[Run]:
    out: list[Run] = []
    for s in spans:
        typo = _typo(s)
        if out and (out[-1].typo == typo or s["text"].isspace()):    # a blank carries no style of its own
            out[-1].text += s["text"]
        else:
            out.append(Run(s["text"], typo))
    return out


def _centre(line: _Line) -> float:
    """Centre of the base-letter ink, the y the OCR path measures and fit_check aligns."""
    biggest = max(line.runs, key=lambda r: r.typo.size)
    return line.baseline + ink_centre_offset(line.text, biggest.typo)


def _inside(rect: fitz.Rect, areas: list[fitz.Rect]) -> bool:
    c = (rect.tl + rect.br) / 2
    return any(c in a for a in areas)


# ---------------------------------------------------------------- tables

def _edges(values: list[float], tol: float = 1.5) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _index(edges: list[float], v: float) -> int:
    return int(np.argmin([abs(e - v) for e in edges]))


def _table(page: fitz.Page, tab) -> TableItem | None:
    cells = [fitz.Rect(c) for c in tab.cells if c]
    if not cells:
        return None
    xs = _edges([c.x0 for c in cells] + [c.x1 for c in cells])
    ys = _edges([c.y0 for c in cells] + [c.y1 for c in cells])
    if len(xs) < 2 or len(ys) < 2:
        return None
    grid: dict[tuple[int, int], tuple] = {}
    sizes, header_sizes, bold_header = [], [], []
    cell_align = {}
    for c in cells:
        r0, r1, c0, c1 = _index(ys, c.y0), _index(ys, c.y1), _index(xs, c.x0), _index(xs, c.x1)
        if r1 <= r0 or c1 <= c0 or (r0, c0) in grid:
            continue
        spans = [s for b in page.get_text("dict", clip=c)["blocks"] for ln in b.get("lines", [])
                 for s in ln["spans"] if s["text"].strip() and fitz.Rect(s["bbox"]).intersects(c)]
        text = " ".join(page.get_textbox(c).split())
        grid[(r0, c0)] = (r1 - r0, c1 - c0, text)
        for s in spans:
            (header_sizes if r0 == 0 else sizes).append((s["size"], len(s["text"])))
            if r0 == 0:
                bold_header.append(_typo(s).bold)
        if spans:
            lx = min(s["bbox"][0] for s in spans) - c.x0
            rx = c.x1 - max(s["bbox"][2] for s in spans)
            cell_align[(r0, c0)] = "center" if abs(lx - rx) < 3 else ("left" if lx < rx else "right")
    rows = []
    for r in range(len(ys) - 1):
        tds = [f'<td rowspan="{rs}" colspan="{cs}">{html.escape(t)}</td>'
               for (rr, c), (rs, cs, t) in sorted(grid.items()) if rr == r]
        rows.append("<tr>" + "".join(tds) + "</tr>")

    def weighted(pairs):
        if not pairs:
            return None
        values = np.repeat([p[0] for p in pairs], [max(1, p[1]) for p in pairs])
        return snap_size(float(np.median(values)))

    col_align = {}
    for col in range(len(xs) - 1):
        votes = [a for (r, c), a in cell_align.items() if c == col and r > 0]
        if votes:
            col_align[col] = max(set(votes), key=votes.count)
    body = weighted(sizes) or weighted(header_sizes) or 12.0
    item = TableItem(xs[0], ys[0], xs[-1], ys[-1], "<table>" + "".join(rows) + "</table>", size=body,
                     col_edges=xs, row_edges=ys, header_size=weighted(header_sizes), cell_align=cell_align,
                     col_align=col_align)
    item.header_bold = bool(bold_header) and sum(bold_header) > len(bold_header) / 2
    return item


# ---------------------------------------------------------------- pictures and rules

def _picture(page: fitz.Page, info: dict, dst: Path) -> Path:
    try:
        pix = fitz.Pixmap(page.parent, info["xref"])
        if pix.colorspace is None or pix.colorspace.n > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        pix.save(str(dst))
    except Exception:                                   # inline image / unusual encoding: render the box
        page.get_pixmap(clip=fitz.Rect(info["bbox"]), dpi=200).save(str(dst))
    return dst


def _rules(page: fitz.Page, tables: list[fitz.Rect], assets: Path) -> list[FloatImage]:
    """Thin stand-alone vector lines (letterhead underlines, signature lines) as clean pictures."""
    out = []
    for n, d in enumerate(page.get_drawings()):
        r = d["rect"]
        thin = min(r.width, r.height)
        if thin > 2.5 or max(r.width, r.height) < 8 or _inside(r, tables):
            continue
        rgb = d.get("color") or d.get("fill") or (0, 0, 0)
        w = max(1, int(r.width * 4))
        h = max(1, int(max(thin, d.get("width") or 0.5) * 4))
        if r.height > r.width:
            w, h = h, w
        png = assets / f"rule_{n}.png"
        Image.new("RGB", (w, h), tuple(int(255 * v) for v in rgb)).save(png)
        half = max(thin, d.get("width") or 0.5) / 2
        if r.height > r.width:
            out.append(FloatImage(r.x0 + r.width / 2 - half, r.y0, r.x0 + r.width / 2 + half, r.y1, png))
        else:
            out.append(FloatImage(r.x0, r.y0 + r.height / 2 - half, r.x1, r.y0 + r.height / 2 + half, png))
    return out


# ---------------------------------------------------------------- text

def _lines_of(block: dict, skip: list[fitz.Rect]) -> list[_Line]:
    lines = []
    for ln in block["lines"]:
        if abs(ln["dir"][1]) > 0.05:                    # rotated text: not handled here
            continue
        for s in ln["spans"]:                           # rawdict: exact extent of the glyphs, not of blanks
            s["text"] = "".join(c["c"] for c in s["chars"])
            ink = [c["bbox"] for c in s["chars"] if not c["c"].isspace()]
            s["ink"] = (min(b[0] for b in ink), max(b[2] for b in ink)) if ink else None
        spans = [s for s in ln["spans"] if s["text"] and not _inside(fitz.Rect(s["bbox"]), skip)]
        visible = [s for s in spans if s["ink"]]
        if not visible:
            continue
        lines.append(_Line(_runs(spans), min(s["ink"][0] for s in visible), max(s["ink"][1] for s in visible),
                           max(visible, key=lambda s: s["size"])["origin"][1], max(s["size"] for s in visible)))
    # pieces PyMuPDF split on one baseline (a tab, a wide space) stay one line when they are close
    lines.sort(key=lambda l: (round(l.baseline), l.x0))
    merged: list[_Line] = []
    for l in lines:
        m = merged[-1] if merged else None
        if m and abs(m.baseline - l.baseline) < 0.3 * l.size and 0 <= l.x0 - m.x1 < 1.5 * l.size:
            m.runs += [Run(" ", m.runs[-1].typo)] + l.runs if not m.text.endswith(" ") else l.runs
            m.x1, m.size = l.x1, max(m.size, l.size)
        else:
            merged.append(l)
    for l in merged:                                   # trailing / leading blanks carry no position
        l.runs[0].text = l.runs[0].text.lstrip()
        l.runs[-1].text = l.runs[-1].text.rstrip()
    return merged


def _flows(lines: list[_Line]) -> bool:
    """Lines of one block form one wrapped paragraph: one per row, same left edge (bar a first-line indent),
    and every break is forced by the width (the next line's first word would not have fitted)."""
    if len(lines) < 2:
        return False
    base = [l.baseline for l in lines]
    pitch = np.diff(base)
    if min(pitch) < 0.8 * max(l.size for l in lines) or max(pitch) > 1.35 * min(pitch):
        return False
    body_x0 = float(np.median([l.x0 for l in lines[1:]]))
    if any(abs(l.x0 - body_x0) > 1.5 for l in lines[1:]):
        return False
    x0s, x1s = [l.x0 for l in lines], [l.x1 for l in lines]
    mids = [(a + b) / 2 for a, b in zip(x0s, x1s)]
    if max(x0s) - min(x0s) > 2 and (max(mids) - min(mids) < 1.5 or max(x1s) - min(x1s) < 1.0):
        return False                                    # centred / right-aligned lines: each placed as is
    right = max(l.x1 for l in lines[:-1])
    for cur, nxt in zip(lines, lines[1:]):
        word = nxt.text.split()[0] if nxt.text.split() else ""
        t = nxt.runs[0].typo
        need = text_width_pt(" " + word, t.size, t.font, t.bold, t.italic)
        if cur.x1 + need < right - 1.0:                 # the word would have fitted: a deliberate break
            return False
    return True


def _paragraph(lines: list[_Line]) -> Para:
    body_x0 = float(np.median([l.x0 for l in lines[1:]]))
    right = max(l.x1 for l in lines[:-1])
    runs: list[Run] = []
    for l in lines:
        if runs:
            prev = runs[-1]
            if not prev.text.endswith(("-", " ")):
                prev.text += " "
        runs += [Run(r.text, r.typo) for r in l.runs]
    size = max(l.size for l in lines)
    first = lines[0].x0 - body_x0
    return Para(body_x0, right + 0.6, [runs], [_centre(l) for l in lines], float(np.median(np.diff(
        [l.baseline for l in lines]))), wrap=True, justify=all(l.x1 > right - 1.0 for l in lines[:-1])
        and len(lines) > 2, first_indent=first if abs(first) > 0.3 * size else 0.0, align="left")


def _single(line: _Line, pitch: float | None) -> Para:
    lh = pitch if pitch and line.size <= pitch <= 2.5 * line.size else line.size * 1.15
    return Para(line.x0, line.x1, [line.runs], [_centre(line)], lh, line_x=[(line.x0, line.x1)])


def extract_page(page: fitz.Page, assets: Path) -> Page:
    assets.mkdir(parents=True, exist_ok=True)
    hidden = [fitz.Rect(s["bbox"]) for s in page.get_texttrace() if s["type"] == 3]
    flow: list = []
    table_rects = []
    try:
        found = page.find_tables().tables
    except Exception:
        found = []
    for tab in found:
        item = _table(page, tab)
        if item is not None:
            flow.append(item)
            table_rects.append(fitz.Rect(tab.bbox))
    skip = hidden + table_rects

    for block in page.get_text("rawdict")["blocks"]:
        if block["type"] != 0:
            continue
        lines = _lines_of(block, skip)
        if not lines:
            continue
        if _flows(lines):
            flow.append(_paragraph(lines))
            continue
        pitches = np.diff([l.baseline for l in lines])
        pitch = float(np.median(pitches)) if len(pitches) else None
        flow += [_single(l, pitch) for l in lines]

    floats = []
    area = page.rect.width * page.rect.height
    for n, info in enumerate(page.get_image_info(xrefs=True)):
        r = fitz.Rect(info["bbox"]) & page.rect
        if r.is_empty or r.width * r.height < 16 or r.width * r.height > 0.95 * area:
            continue
        floats.append(FloatImage(r.x0, r.y0, r.x1, r.y1, _picture(page, info, assets / f"img_{n}.png")))
    floats += _rules(page, table_rects, assets)
    return Page(page.rect.width, page.rect.height, flow, floats)
