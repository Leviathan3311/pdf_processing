"""Turn MinerU middle.json + the page image into positioned, typed layout items (CLAUDE.md section 7.1).

Units: PDF points, origin top-left of the page. Page size comes from the image aspect ratio.

Items
  Para        printed text; one entry per text line slot with measured centre, size, weight, slant, colour
  TableItem   real table: HTML grid (row/col spans), column/row lines measured on the image
  FloatImage  coloured ink (handwriting, signature) and MinerU image blocks (seals, photos): cropped from
              the scan with a transparent background, placed at their exact position
Flow items (Para, TableItem) are arranged by recursive XY-cut into VStack / HSplit nodes; HSplit becomes a
borderless table in Word so side-by-side blocks (two-column letterhead, "Nơi nhận" | signature) stay put.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .measure import (Typo, cluster_sizes, fit_paragraph, fit_single_line, ink_mask, is_bold, measure_block,
                      split_by_widths)

A4 = (595.3, 841.9)


@dataclass
class Run:
    text: str                       # shown text (for a formula: what it displays, used for widths / matching)
    typo: Typo
    latex: str | None = None        # set for a formula: rendered as a native Word equation


@dataclass
class Para:
    x0: float
    x1: float
    lines: list[list[Run]]          # forced lines (one entry per visual line slot for header-like blocks)
    centres: list[float]            # measured centre y of each visual line (pt)
    line_height: float              # exact line height (pt)
    wrap: bool = False              # True: one flowing paragraph whose text wraps (body text)
    justify: bool = False
    first_indent: float = 0.0
    align: str = "center"           # for single / forced lines: left | center | right
    outline: int | None = None      # heading level for navigation
    target: float | None = None     # measured centre of the first line on the scan (fit_check keeps it fixed)
    line_x: list[tuple[float, float]] | None = None   # per-line printed extent (forced lines are placed
                                                      # individually: real letterheads are not perfectly centred)
    tab_stops: list[float] = field(default_factory=list)   # absolute x (pt) of grid columns ("a) …   b) …")
    stack: float = 1.0              # tallest formula on the line, in text rows (a 3-row matrix ≈ 3.5)
    ink: tuple[float, float] | None = None   # measured horizontal extent before widening to the typeset text:
                                             # what the XY-cut sees (widened boxes can close a column gap)

    @property
    def y0(self) -> float:
        return self.centres[0] - self.line_height / 2

    @property
    def y1(self) -> float:
        return self.centres[-1] + self.line_height / 2


@dataclass
class TableItem:
    x0: float
    y0: float
    x1: float
    y1: float
    html: str
    size: float = 12.0
    col_edges: list[float] | None = None       # x of vertical rules (pt), len = ncols + 1
    row_edges: list[float] | None = None       # y of horizontal rules (pt), len = nrows + 1
    borders: bool = True
    header_size: float | None = None                  # first row, measured separately (often bold, smaller)
    cell_align: dict = field(default_factory=dict)   # (row, col) -> left | center | right, measured
    col_align: dict = field(default_factory=dict)    # col -> majority alignment of its body cells
    header_bold: bool = True                          # first row bold (scans: assumed; PDFs: read)


@dataclass
class FloatImage:
    x0: float
    y0: float
    x1: float
    y1: float
    png: Path


@dataclass
class Page:
    width: float
    height: float
    flow: list                      # Para | TableItem
    floats: list[FloatImage] = field(default_factory=list)


# ---------------------------------------------------------------- page geometry

def page_size(img_w: int, img_h: int) -> tuple[float, float]:
    ratio = img_h / img_w
    if abs(ratio - A4[1] / A4[0]) < 0.06:
        return A4
    if abs(ratio - A4[0] / A4[1]) < 0.04:
        return A4[1], A4[0]
    return A4[0], A4[0] * ratio


# ---------------------------------------------------------------- middle.json helpers

EQUATION_TYPES = {"equation", "equation_inline", "inline_equation", "interline_equation"}


def _segments(content, kind: str = "text") -> list[tuple[str, str]]:
    """Ordered ('text' | 'math', value) pieces of a block; LaTeX stays separate from prose."""
    if isinstance(content, str):
        return [("math" if kind in EQUATION_TYPES else "text", content)]
    out: list[tuple[str, str]] = []
    for item in content or []:
        if isinstance(item, dict) and item.get("type") not in ("image_body", "table_body"):
            out += _segments(item.get("content"), item.get("type", "text"))
    return out


Token = tuple  # (kind: 'text' | 'math' | 'br', value, space_before)


def _tokens(segs: list[tuple[str, str]]) -> list[Token]:
    """Words and formulas in reading order, remembering whether a space preceded each; '\\n' → 'br'."""
    tokens: list[Token] = []
    pending_space = False
    for kind, value in segs:
        if kind == "math":
            if value.strip():
                tokens.append(("math", value.strip(), pending_space))
            pending_space = False
            continue
        for i, part in enumerate(value.split("\n")):
            if i:
                tokens.append(("br", "", False))
                pending_space = False
            words = part.split()
            for j, word in enumerate(words):
                space = pending_space or j > 0 or (j == 0 and part[:1].isspace())
                tokens.append(("text", word, space))
            if words:
                pending_space = part[-1:].isspace()
            elif part:
                pending_space = True
    return tokens


def _token_lines(tokens: list[Token]) -> list[list[Token]]:
    lines, cur = [], []
    for t in tokens:
        if t[0] == "br":
            if cur:
                lines.append(cur)
            cur = []
        else:
            cur.append(t)
    return lines + ([cur] if cur else [])


def _shown(token: Token) -> str:
    if token[0] == "math":
        from .math_omml import display_text
        return display_text(token[1])
    return token[1]


def _plain(tokens: list[Token]) -> str:
    """What the tokens look like on paper: used for width measurement and size fitting."""
    out = ""
    for t in tokens:
        out += (" " if out and t[2] else "") + _shown(t)
    return out


def _runs(tokens: list[Token], typo: Typo) -> list[Run]:
    runs: list[Run] = []
    for t in tokens:
        if t[0] == "math":
            if t[2] and runs and not runs[-1].latex:
                runs[-1].text += " "
            elif t[2] and runs:
                runs.append(Run(" ", typo))
            runs.append(Run(_shown(t), typo, latex=t[1]))
        elif runs and not runs[-1].latex:
            runs[-1].text += (" " if t[2] else "") + t[1]
        else:
            runs.append(Run((" " if t[2] and runs else "") + t[1], typo))
    return runs


def _unit_width(tokens: list[Token], bold: bool = False, italic: bool = False) -> float:
    """Width (pt) of the tokens at 1 pt: prose in Times New Roman, formulas in Cambria Math."""
    from .math_omml import width_em
    from .measure import text_width_pt
    width = 0.0
    for i, t in enumerate(tokens):
        if i and t[2]:
            width += text_width_pt(" ", 1.0, bold=bold, italic=italic)
        width += width_em(t[1]) if t[0] == "math" else text_width_pt(t[1], 1.0, bold=bold, italic=italic)
    return width


def _match_lines(units: list[float], widths: list[float], gaps: list[float]) -> list[tuple]:
    """Map MinerU's lines (split at '\\n', unit widths at 1 pt) onto the lines measured on the scan.

    Each step either lets one MinerU line span k ≥ 1 scan lines (text wrapping) or puts several MinerU lines
    on one scan line (exercise grids "a) …   b) …", whose column gaps are measured as blank runs).
    Dynamic programming over the shared, unknown font size; returns [('span', i, j0, j1) | ('group', i0, i1, j)].
    """
    n, m = len(units), len(widths)
    best_total, best_plan = float("inf"), None
    for size in np.arange(6.0, 30.01, 0.5):
        cost = {(0, 0): (0.0, None)}
        for i in range(n + 1):
            for j in range(m + 1):
                if (i, j) not in cost or (i == n and j == m):
                    continue
                base = cost[(i, j)][0]
                for k in range(1, m - j + 1) if i < n else ():              # one line over k scan lines
                    w = sum(widths[j:j + k])
                    if k == 1:
                        err = abs(units[i] * size - w) / w
                    else:                                    # wrapped: fills all but the last line
                        err = (max(0.0, units[i] * size - w)
                               + max(0.0, sum(widths[j:j + k - 1]) - units[i] * size)) / w
                    key, val = (i + 1, j + k), (base + err, ("span", i, j, j + k - 1))
                    if key not in cost or val[0] < cost[key][0]:
                        cost[key] = val
                for g in range(2, n - i + 1) if j < m else ():                # g lines on one scan line
                    w = widths[j]
                    err = abs(sum(units[i:i + g]) * size + gaps[j] - w) / w + 0.05
                    key, val = (i + g, j + 1), (base + err, ("group", i, i + g - 1, j))
                    if key not in cost or val[0] < cost[key][0]:
                        cost[key] = val
        if (n, m) in cost and cost[(n, m)][0] < best_total:
            best_total, best_plan = cost[(n, m)][0], cost
    if best_plan is None:
        return []
    plan, state = [], (n, m)
    while state != (0, 0):
        step = best_plan[state][1]
        plan.append(step)
        state = (step[1], step[2]) if step[0] == "span" else (step[1], step[3])
    return plan[::-1]


def _pieces(ln, page_w_px: float | None = None) -> list:
    """A measured line cut at a column gap: 'VĂN PHÒNG CHÍNH PHỦ   |   CỘNG HÒA XÃ HỘI …' → 2 pieces.

    Measured on four letters: the column gap is the widest blank near the middle of the page, 1.3–7× the
    line height and ≥ 4× the line's median blank, while word gaps, spaced or faint capitals and blank
    fill-in fields ("ngày …… tháng") are ≤ 1.4× or lie inside one half. Hence: split once, at the widest
    blank centred in 30–70 % of the page width, if it is ≥ 1× line height and ≥ 3× the median blank."""
    from dataclasses import replace
    h = ln.letter_h or (ln.y1 - ln.y0)           # letter height: stacked diacritics inflate the line extent
    if page_w_px is None or not ln.blanks:
        return [ln]
    middle = [(s, e) for s, e in ln.blanks if 0.3 * page_w_px <= (s + e) / 2 <= 0.7 * page_w_px]
    if not middle:
        return [ln]
    s, e = max(middle, key=lambda g: g[1] - g[0])
    median_blank = float(np.median([b - a for a, b in ln.blanks]))
    if e - s < max(1.0 * h, 3.0 * median_blank):
        return [ln]
    pieces = [replace(ln, x1=s, gaps=[], blanks=[]), replace(ln, x0=e, gaps=[], blanks=[])]
    pieces = [p for p in pieces if p.x1 - p.x0 > 1.2 * h]             # drop slivers (left-over rule ink)
    if len(pieces) == 2:
        # two page columns: one piece on each side of the page centre (a gap inside a line that sits in
        # one half — a hand-filled date, a wrapped subject line — is not a column gap)
        left_c, right_c = [(p.x0 + p.x1) / 2 for p in pieces]
        if not (left_c < 0.5 * page_w_px < right_c):
            return [ln]
    return pieces or [ln]


def _text_paras(tokens: list[Token], m, box: tuple[float, float], sx: float, sy: float, median_dark: float,
                title: bool, outline, standalone_lines: bool = False,
                page_w_px: float | None = None, read_lines: bool = False) -> list["Para"]:
    """Paragraphs for one text block: measured lines, sizes, weights, and how MinerU's lines sit on them.

    `standalone_lines`: the block is a heading / letterhead / footer — its lines never flow as one paragraph.
    `read_lines`: handwriting (read by the VLM, or any block of a pen page without a matrix): the reader's line
    breaks are the written lines, more reliable than matching uneven handwriting by width.
    Otherwise a MinerU text line spanning several scan lines flows (wraps) unless the lines are short and
    centred (a signature block): MinerU's block type is a sturdier signal than line widths, which bend on
    curled or photographed pages."""
    from .measure import is_bold
    lines = m.lines
    forced = _token_lines(tokens)
    block_w = box[1] - box[0]

    def bold_of(ln) -> bool:
        verdict = is_bold(ln.darkness, median_dark)
        return title if verdict is None else verdict

    def centre(ln) -> float:
        return (ln.y0 + ln.y1) / 2 * sy

    def single(line_tokens, ln, tabs=None) -> "Para":
        bold = bold_of(ln)
        width = (ln.x1 - ln.x0) * sx
        if tabs:                                       # grid line: column gaps are white space, not text
            width -= sum(e - s for s, e in ln.gaps) * sx
        unit = _unit_width(line_tokens, bold, m.italic)
        typo = Typo(size=width / unit if unit else 12.0, bold=bold, italic=m.italic)
        runs = _runs(line_tokens, typo)
        tall = _stack_em(line_tokens)                  # a fraction / matrix needs more than one text row
        para = Para(ln.x0 * sx, ln.x1 * sx, [runs], [centre(ln)], (ln.y1 - ln.y0) * sy if tall > 1.5 else 0.0,
                    outline=outline, line_x=[(ln.x0 * sx, ln.x1 * sx)])
        para.stack = tall
        para.tab_stops = tabs or []
        if tabs:
            para.align = "left"
        return para

    def flowing(line_tokens, lns) -> "Para":
        bold = sum(bold_of(ln) for ln in lns) > len(lns) / 2
        centres = [centre(ln) for ln in lns]
        pitch = float(np.median(np.diff(centres)))
        body_x0 = float(np.median([ln.x0 for ln in lns[1:]])) * sx
        body_x1 = max(ln.x1 for ln in lns[:-1]) * sx
        first_indent = lns[0].x0 * sx - body_x0
        size = fit_paragraph(_plain(line_tokens), body_x1 - body_x0, len(lns), (lns[-1].x1 - lns[-1].x0) * sx,
                             bold, m.italic, pitch_pt=pitch, first_indent=max(0.0, first_indent),
                             line_widths=[(ln.x1 - ln.x0) * sx for ln in lns])
        body = [(ln.x1 - ln.x0) * sx for ln in lns[:-1]]
        return Para(body_x0, body_x1, [_runs(line_tokens, Typo(size=size, bold=bold, italic=m.italic))],
                    centres, pitch or size * 1.2, wrap=True, justify=min(body) > 0.93 * block_w,
                    first_indent=first_indent if abs(first_indent) > 0.3 * size else 0.0,
                    align="left", outline=outline)

    if read_lines and len(forced) > 1:
        # the handwriting reader's line breaks are the written lines. Matching them onto the measured lines
        # by width fails on handwriting (uneven letters, lines that touch, drift or curl) and sets stray words
        # huge, so: one size for the block that fits the longest line and the line pitch, left-aligned; each
        # line on its measured line when the counts agree, else the block's ink height shared evenly
        top, bottom = min(ln.y0 for ln in lines) * sy, max(ln.y1 for ln in lines) * sy
        left, right = min(ln.x0 for ln in lines) * sx, max(ln.x1 for ln in lines) * sx
        bold = sum(bold_of(ln) for ln in lines) > len(lines) / 2
        # size from all the ink against all the text, not from the longest line: the reader sometimes joins two
        # written lines into one; such a line is split again at the block width
        units = [_unit_width(f, bold, m.italic) for f in forced]
        ink_width = sum((ln.x1 - ln.x0) * sx for ln in lines)
        size = ink_width / sum(units) if sum(units) > 0 else 12.0
        size = min(size, (bottom - top) / max(len(lines), len(forced)) / 1.15)
        split: list[list[Token]] = []
        for f, u in zip(forced, units):
            k = int(np.ceil(u * size / (1.05 * (right - left)))) if u > 0 else 1
            split += _split_tokens(f, [right - left] * k) if k > 1 else [f]
        forced = [f for f in split if f]
        units = [_unit_width(f, bold, m.italic) for f in forced]
        pitch = (bottom - top) / len(forced)
        same = len(forced) == len(lines)
        centres = ([centre(ln) for ln in lines] if same
                   else [top + pitch * (k + 0.5) for k in range(len(forced))])
        starts = [ln.x0 * sx for ln in lines] if same else [left] * len(forced)
        fits = min((right - s) / u for s, u in zip(starts, units) if u > 0) if max(units) > 0 else size
        size = min(size, fits, pitch / 1.15)
        typo = Typo(size=size, bold=bold, italic=m.italic)      # shared: the block's size is set once
        paras = []
        for f, c, s, u in zip(forced, centres, starts, units):   # a line each, so only a fraction's line
            para = Para(s, s + u * size, [_runs(f, typo)], [c], pitch, align="left", outline=outline,
                        line_x=[(s, s + u * size)])              # grows taller
            para.stack = _stack_em(f)
            paras.append(para)
        return paras

    widths = [(ln.x1 - ln.x0) * sx for ln in lines]
    gaps = [sum(e - s for s, e in ln.gaps) * sx for ln in lines]
    plan = _match_lines([_unit_width(f) for f in forced], widths, gaps)
    if not plan:                                       # could not map: keep the whole block as one line set
        plan = [("span", 0, 0, len(lines) - 1)]
        forced = [[t for f in forced for t in f]]
    paras: list[Para] = []
    for step in plan:
        if step[0] == "group":
            _, i0, i1, j = step
            ln = lines[j]
            columns = len(ln.gaps) >= i1 - i0                # a real grid: one measured gap per extra column
            joined: list[Token] = []
            for k, f in enumerate(forced[i0:i1 + 1]):
                if k and columns:
                    joined.append(("tab", "\t", False))
                joined += [(f[0][0], f[0][1], True)] + f[1:] if k and not columns and f else f
            tabs = [e * sx for s, e in sorted(ln.gaps, key=lambda g: g[1] - g[0], reverse=True)[:i1 - i0]]
            paras.append(single(joined, ln, sorted(tabs) if columns else None))
            continue
        _, i, j0, j1 = step
        lns = lines[j0:j1 + 1]
        widths_here = [(ln.x1 - ln.x0) * sx for ln in lns]
        mids = [(ln.x0 + ln.x1) / 2 * sx for ln in lns]
        centred_short = max(widths_here) < 0.8 * block_w and max(mids) - min(mids) < 0.06 * block_w
        # centred lines (a subject line "V/v …" under the number): left edges wander, centres stay put
        h = float(np.median([ln.letter_h for ln in lns])) * sx or 1.0
        x0_spread = (max(ln.x0 for ln in lns) - min(ln.x0 for ln in lns)) * sx
        centred_short = centred_short or (x0_spread > 0.75 * h and max(mids) - min(mids) < 0.3 * x0_spread)
        # a wrapped line holds several words; a signer's name spread over noise "lines" (a seal) does not wrap
        enough_words = sum(1 for t in forced[i] if t[0] != "tab") >= 4 * len(lns)
        if len(lns) > 1 and not standalone_lines and not centred_short and enough_words:
            paras.append(flowing(forced[i], lns))
        elif len(lns) > 1 and _stack_em(forced[i]) > 1.5:
            # one MinerU line holding a matrix / stacked fraction measures as several rows: it is one line
            paras.append(single(forced[i], _union(lns)))
        else:                                          # independent lines; a line with a wide gap is two
            pieces = [p for ln in lns for p in _pieces(ln, page_w_px)]   # (merged two-column letterhead)
            words = sum(1 for t in forced[i] if t[0] != "tab")
            side_by_side = sum(p.x1 - p.x0 for p in pieces) <= 1.1 * (max(p.x1 for p in pieces)
                                                                     - min(p.x0 for p in pieces))
            if 1 < len(pieces) and words < len(pieces) and side_by_side:
                # fewer words than pieces lying one after another: a slanted handwritten line measured in
                # fragments, not separate lines — keep it whole instead of squeezing it into one fragment
                paras.append(single(forced[i], _union(lns)))
                continue
            parts = _split_tokens(forced[i], [(p.x1 - p.x0) * sx for p in pieces])
            for part, piece in zip(parts, pieces):
                paras.append(single(part, piece))
    return paras


def _stack_em(tokens: list[Token]) -> float:
    """Tallest formula among the tokens, in em (1 = one text row)."""
    from .math_omml import height_em
    return max((height_em(t[1]) for t in tokens if t[0] == "math"), default=1.0)


def _union(lns: list) -> "Line":
    """Measured rows that make up one tall line (matrix rows, fraction parts) as one line."""
    from .measure import Line
    return Line(y0=min(l.y0 for l in lns), y1=max(l.y1 for l in lns), x0=min(l.x0 for l in lns),
                x1=max(l.x1 for l in lns), density=float(np.mean([l.density for l in lns])),
                darkness=float(np.median([l.darkness for l in lns])),
                letter_h=float(np.median([l.letter_h for l in lns])))


def _split_tokens(tokens: list[Token], widths: list[float]) -> list[list[Token]]:
    """Distribute tokens (reading order) over measured line pieces (same order).

    Every piece implies a font size: measured width / typeset width of its words. A document uses few sizes,
    so the split keeps those implied sizes as close to each other as possible (least squares on log size,
    dynamic programming over the break points, mean re-estimated). Rounding each break independently by
    width share lets a word slip into the next piece when the pieces are set at different sizes."""
    from .measure import text_width_pt
    k = len(widths)
    if k <= 1 or len(tokens) <= 1:
        return [tokens]
    if len(tokens) < k:                                # fewer words than pieces: one word each
        return [[t] for t in tokens]
    unit = np.array([text_width_pt(_shown(t) + " ", 1.0) for t in tokens])
    cum = np.concatenate([[0.0], np.cumsum(unit)])
    n = len(tokens)
    mu = np.log(sum(widths) / cum[-1])
    for _ in range(3):
        cost = np.full((k + 1, n + 1), np.inf)
        back = np.zeros((k + 1, n + 1), int)
        cost[0, 0] = 0.0
        for i in range(1, k + 1):
            for j in range(i, n - (k - i) + 1):
                prev = np.arange(i - 1, j)
                seg = cum[j] - cum[prev]
                err = (np.log(widths[i - 1] / seg) - mu) ** 2 * widths[i - 1]
                total = cost[i - 1, prev] + err
                best = int(np.argmin(total))
                cost[i, j], back[i, j] = total[best], prev[best]
        cuts, j = [], n
        for i in range(k, 0, -1):
            cuts.append(j)
            j = back[i, j]
        cuts = [0] + cuts[::-1]
        logs = [np.log(widths[i] / (cum[cuts[i + 1]] - cum[cuts[i]])) for i in range(k)]
        mu = float(np.average(logs, weights=widths))
    return [tokens[cuts[i]:cuts[i + 1]] for i in range(k)]


def _flatten(blocks: list[dict]) -> list[dict]:
    """List blocks carry their items as children with their own bbox: treat each item as a block."""
    out = []
    for b in blocks:
        children = [c for c in (b.get("content") or []) if isinstance(c, dict) and c.get("bbox")
                    and c.get("type") not in ("table_body", "image_body")]
        if b.get("type") == "list" and children:
            out += [{**c, "type": "text"} for c in children]
        elif b.get("type") == "table":
            out.append(b)
            out += [{**c, "type": "text"} for c in children]       # captions / footnotes with own bbox
        else:
            out.append(b)
    return out


# ---------------------------------------------------------------- image helpers

def _crop_png(rgb: np.ndarray, box: tuple[int, int, int, int], dst: Path, ink_ref=None,
              text_boxes: list | None = None) -> Path:
    """Crop with transparent paper so the float does not hide anything under it.

    For pen / seal ink (`ink_ref` given and the crop is mostly coloured ink), printed strokes inside the
    crop are made transparent too where they belong to a text line that is typeset already ("BỘ TRƯỞNG"
    under a seal: a second copy would print it twice). Dark strokes outside every text line (a signature
    in black ink over the seal) stay in the picture. `text_boxes`: page-pixel line boxes; None = all."""
    from .measure import hue_offset
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    alpha = np.clip((235 - gray.astype(int)) * 4, 0, 255).astype(np.uint8)   # paper → transparent
    if ink_ref is not None and crop.size:
        ink = gray < 200
        coloured = hue_offset(crop, ink_ref) >= 16
        if ink.any() and (coloured & ink).sum() > 0.3 * ink.sum():
            typeset = np.ones_like(ink) if text_boxes is None else np.zeros_like(ink)
            for tx0, ty0, tx1, ty1 in text_boxes or []:
                pad = 0.3 * (ty1 - ty0)                  # diacritics sit above the measured letters
                a0, a1 = int(max(ty0 - pad - y0, 0)), int(max(ty1 + pad - y0, 0))
                b0, b1 = int(max(tx0 - x0, 0)), int(max(tx1 - x0, 0))
                typeset[a0:a1, b0:b1] = True
            alpha[ink & ~coloured & typeset] = 0
    rgba = np.dstack([crop, alpha])
    cv2.imwrite(str(dst), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return dst


def line_mask(gray: np.ndarray) -> np.ndarray:
    """Faint thin rules survive upscaling as light grey: use a local (adaptive) threshold for them."""
    block = max(15, (min(gray.shape) // 20) | 1)
    return (cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 12) > 0
            ).astype(np.uint8)


def _rules(gray: np.ndarray, axis: str) -> list[int]:
    """Positions (px) of long ruling lines inside a table crop."""
    mask = line_mask(gray)
    h, w = mask.shape
    if axis == "v":
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 3)))
        prof = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k).sum(axis=0)
        limit = 0.3 * h
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 3), 1))
        prof = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k).sum(axis=1)
        limit = 0.3 * w
    idx = np.where(prof > limit)[0]
    peaks, start = [], None
    for i in list(idx) + [None]:
        if start is None:
            start = prev = i
        elif i is None or i - prev > 3:
            peaks.append((start + prev) // 2)
            start = prev = i
        else:
            prev = i
    return [p for p in peaks if p is not None]


def _grid_shape(html: str) -> tuple[int, int]:
    from lxml import html as lh
    rows = lh.fromstring(html).xpath("//tr")
    occupied: dict[tuple[int, int], bool] = {}
    ncols = 0
    for r, tr in enumerate(rows):
        c = 0
        for cell in tr.xpath("./td|./th"):
            while occupied.get((r, c)):
                c += 1
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            for dr in range(rs):
                for dc in range(cs):
                    occupied[(r + dr, c + dc)] = True
            c += cs
        ncols = max(ncols, c)
    return len(rows), ncols


# ---------------------------------------------------------------- build

def build_page(middle_page: dict, image_path: Path, assets: Path,
               page_pt: tuple[float, float] | None = None) -> Page:
    """`page_pt`: the real page size when known (a PDF page); otherwise inferred from the image shape."""
    bgr = cv2.imread(str(image_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    pw, ph = page_pt or page_size(img_w, img_h)
    sx, sy = pw / img_w, ph / img_h
    assets.mkdir(parents=True, exist_ok=True)

    def px(b):
        return (int(b[0] * img_w), int(b[1] * img_h), int(b[2] * img_w), int(b[3] * img_h))

    blocks = sorted(_flatten(middle_page.get("blocks", [])), key=lambda b: b.get("index", 0))

    # pass 1: measure every text block (typical line height first, to fold stray accent groups)
    from .math_omml import height_em

    def has_math(b) -> bool:
        # a formula taller than the text (scripts 1.3 em, fractions, matrices) splits its line into stacked
        # groups; a symbol on the line (→, k, softmax) does not, and must not switch off the text rules
        return math_em(b) > 1.0

    def math_em(b) -> float:
        return max((height_em(v) for k, v in _segments(b.get("content"), b["type"]) if k == "math"), default=0.0)

    from .measure import ink_reference, is_pen_page
    text_boxes_px = [px(b["bbox"]) for b in blocks if b["type"] not in ("image", "table") and b.get("bbox")]
    ink_ref = ink_reference(rgb, gray, text_boxes_px)   # the page's own print colour, from its text only
    pen_page = is_pen_page(ink_ref)                     # handwritten page: keep every OCR'd line as text
    measured = []
    for b in blocks:
        if b["type"] in ("image", "table") or not b.get("bbox"):
            measured.append(None)
            continue
        measured.append(measure_block(rgb, gray, px(b["bbox"]), has_math=has_math(b), ink_ref=ink_ref,
                                      pen_page=pen_page, tall_math=math_em(b) > 2.0))
    heights = [ln.y1 - ln.y0 for m in measured if m for ln in m.lines]
    ref_h = float(np.median(heights)) if heights else None
    measured = [measure_block(rgb, gray, px(b["bbox"]), ref_h, has_math=has_math(b), ink_ref=ink_ref,
                              pen_page=pen_page, tall_math=math_em(b) > 2.0)
                if m is not None else None for b, m in zip(blocks, measured)]
    if pen_page:                                        # a red heading among blue notes is still the text
        for m in measured:
            if m is not None:
                m.colored_ink = False
    dark = [ln.darkness for m in measured if m and not m.colored_ink for ln in m.lines]
    median_dark = float(np.median(dark)) if dark else 128.0

    line_boxes = [(ln.x0, ln.y0, ln.x1, ln.y1) for m in measured if m and not m.colored_ink for ln in m.lines]
    flow, floats, size_slots = [], [], []      # size_slots: (Typo) objects to cluster afterwards
    read_typos: list[Typo] = []                # one per multi-line block set from its read lines
    for n, (b, m) in enumerate(zip(blocks, measured)):
        box = px(b["bbox"])
        x0, y0, x1, y1 = b["bbox"][0] * pw, b["bbox"][1] * ph, b["bbox"][2] * pw, b["bbox"][3] * ph
        # coloured ink stays a picture unless the handwriting reader turned it into text (DESIGN 5.4)
        if b["type"] == "image" or (m is not None and m.colored_ink and not b.get("hw")):
            floats.append(FloatImage(x0, y0, x1, y1, _crop_png(rgb, box, assets / f"float_{n}.png", ink_ref,
                                                                  line_boxes)))
            continue
        if b["type"] == "table":
            html = next((c.get("content") for c in b.get("content", []) if c.get("type") == "table_body"), "")
            if not html:
                continue
            pad = max(4, img_w // 200)                    # MinerU's box can sit just inside the outer rules
            cx0, cy0 = max(0, box[0] - pad), max(0, box[1] - pad)
            crop = gray[cy0:box[3] + pad, cx0:box[2] + pad]
            nrows, ncols = _grid_shape(html)
            vs, hs = _rules(crop, "v"), _rules(crop, "h")
            col_edges = [(cx0 + v) * sx for v in vs] if len(vs) == ncols + 1 else None
            if len(hs) == nrows + 1:
                row_edges = [(cy0 + h) * sy for h in hs]
            else:                                         # some rules not found: share the table height,
                near = 0.03 * (y1 - y0)                   # using found outer rules only if they are outer
                top = (cy0 + hs[0]) * sy if hs and abs((cy0 + hs[0]) * sy - y0) < near else y0
                bottom = (cy0 + hs[-1]) * sy if hs and abs((cy0 + hs[-1]) * sy - y1) < near else y1
                row_edges = [top + (bottom - top) * k / nrows for k in range(nrows + 1)]
            item = TableItem(x0, y0, x1, y1, html, col_edges=col_edges, row_edges=row_edges,
                             borders=bool(vs or hs))
            item.size = _measure_table_text(item, rgb, gray, sx, sy) or 0.0
            flow.append(item)
            continue
        tokens = _tokens(_segments(b.get("content"), b["type"]))
        if not _token_lines(tokens) or m is None or not m.lines:
            continue
        title = b["type"] in ("doc_title", "paragraph_title", "title")
        outline = b.get("level") if title else None
        standalone = title or b["type"] in ("header", "footer", "page_number", "page_footnote")
        read = bool(b.get("hw")) or (pen_page and math_em(b) <= 2.0)
        for para in _text_paras(tokens, m, (x0, x1), sx, sy, median_dark, title, outline, standalone, img_w,
                                read_lines=read):
            for line in para.lines:
                size_slots += [r.typo for r in line]
            flow.append(para)
            typo = para.lines[0][0].typo
            if read and len(_token_lines(tokens)) > 1 and all(typo is not t for t in read_typos):
                read_typos.append(typo)

    # one hand writes at one size: a block whose ink is wider than its text (crossed-out words, a drawn
    # bracket) must not come out far larger than the rest of the page (measured: 2.6 vs 1.3–2.0 elsewhere)
    if len(read_typos) >= 3:
        usual = float(np.median([t.size for t in read_typos]))
        for t in read_typos:
            t.size = min(max(t.size, 0.85 * usual), 1.1 * usual)

    # a "word" of one or two characters set far larger than the page's text is a pen mark ("/", a tick)
    # that OCR read as text: keep it as the picture it is
    sizes = [r.typo.size for it in flow if isinstance(it, Para) for line in it.lines for r in line]
    typical = float(np.median(sizes)) if sizes else 12.0
    marks = [it for it in flow if isinstance(it, Para) and not it.wrap
             and len("".join(r.text for line in it.lines for r in line).strip()) <= 2
             and max(r.typo.size for line in it.lines for r in line) > 1.8 * typical]
    for n, it in enumerate(marks):
        flow.remove(it)
        half = 0.6 * max(r.typo.size for line in it.lines for r in line)     # line height is not set yet
        y0, y1 = it.centres[0] - half, it.centres[-1] + half
        box = (int(it.x0 / sx), max(0, int(y0 / sy)), int(it.x1 / sx) + 1, int(y1 / sy) + 1)
        floats.append(FloatImage(it.x0, max(0.0, y0), it.x1, y1, _crop_png(rgb, box, assets / f"mark_{n}.png")))
        size_slots = [t for t in size_slots if all(t is not r.typo for line in it.lines for r in line)]

    # one document uses one body size: snap flowing paragraphs to the dominant one
    wrap_sizes = [it.lines[0][0].typo.size for it in flow if isinstance(it, Para) and it.wrap]
    if wrap_sizes:
        dominant = float(np.median(wrap_sizes))
        for it in flow:
            if isinstance(it, Para) and it.wrap and abs(it.lines[0][0].typo.size - dominant) <= 2.0:
                it.lines[0][0].typo.size = dominant

    # one document uses one first-line indent: a pen stroke over a paragraph start must not change it
    indents = [it.first_indent for it in flow if isinstance(it, Para) and it.wrap and it.first_indent > 0]
    if indents:
        common = float(np.median(indents))
        for it in flow:
            if isinstance(it, Para) and it.wrap and it.first_indent > 0 and abs(it.first_indent - common) < 0.6 * common:
                it.first_indent = common

    text_boxes = [(ln.x0, ln.y0, ln.x1, ln.y1) for m in measured if m and not m.colored_ink for ln in m.lines]
    floats += _decorative_rules(gray, rgb, [it for it in flow if isinstance(it, TableItem)], text_boxes,
                                sx, sy, assets)

    # documents use few sizes: pull noisy per-line estimates together
    for typo, size in zip(size_slots, cluster_sizes([t.size for t in size_slots])):
        typo.size = size
    for item in flow:
        if isinstance(item, Para) and not item.wrap:
            biggest = max(r.typo.size for line in item.lines for r in line)
            # an exact line height clips a formula taller than one row (sizes are final only here)
            item.line_height = max(item.line_height, biggest * (1.25 * item.stack if item.stack > 1.5 else 1.15))
            if len(item.lines) == 1 and item.line_height == 0.0:
                item.line_height = biggest * 1.15
            _fit_extent(item)
            room = ph - 1.5 - item.centres[-1]             # a signature line at the very bottom of the scan
            if item.line_height / 2 > room:               # must not push onto a second page
                item.line_height = max(0.95 * biggest, 2 * room)
    body_sizes = [r.typo.size for it in flow if isinstance(it, Para) and it.wrap for line in it.lines for r in line]
    body = float(np.median(body_sizes)) if body_sizes else 13.0
    for item in flow:
        if isinstance(item, TableItem):
            # measured cell text when the grid is known, never larger than what fits the columns
            from .measure import snap_size                # Word stores sizes in half points
            item.size = snap_size(min(item.size or body, _table_size(item, item.size or body)))
            if item.header_size:
                item.header_size = snap_size(item.header_size)
    return Page(pw, ph, flow, floats)


# Word sets formulas a little wider than the structural estimate (operator spacing, script gaps): room
# reserved for a standalone line, so its equation is never broken onto a second line
MATH_ROOM = 1.12


def typeset_width(runs: list[Run]) -> float:
    """Width (pt) Word will give one line of runs: Times New Roman text, Cambria Math formulas."""
    from .math_omml import width_em
    from .measure import text_width_pt
    return sum(width_em(r.latex) * r.typo.size * MATH_ROOM if r.latex
               else text_width_pt(r.text, r.typo.size, r.typo.font, r.typo.bold, r.typo.italic) for r in runs)


def _fit_extent(para: Para) -> None:
    """A standalone line keeps its measured centre but is at least as wide as its typeset text, so a line
    partly hidden under a seal or printed faintly never gets a box (or an XY-cut column) narrower than it."""
    boxes = para.line_x or [(para.x0, para.x1)]
    grown = []
    for (x0, x1), line in zip(boxes, para.lines):
        need = typeset_width(line) * 1.04
        pad = max(0.0, (need - (x1 - x0)) / 2)
        grown.append((x0 - pad, x1 + pad))
    para.ink = (para.x0, para.x1)
    para.line_x = grown
    para.x0, para.x1 = min(a for a, _ in grown), max(b for _, b in grown)


def _cells(html: str):
    """(row, col, rowspan, colspan, text) of every cell, following row/col spans."""
    from lxml import html as lh
    taken, out = set(), []
    for r, tr in enumerate(lh.fromstring(html).xpath("//tr")):
        c = 0
        for cell in tr.xpath("./td|./th"):
            while (r, c) in taken:
                c += 1
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, c + dc))
            out.append((r, c, rs, cs, cell.text_content().strip()))
            c += cs
    return out


def _measure_table_text(item: TableItem, rgb, gray, sx: float, sy: float) -> float | None:
    """Font size of the table text, measured inside single cells of the detected grid."""
    if not item.col_edges or not item.row_edges:
        return None
    sizes = []
    for r, c, rs, cs, text in _cells(item.html):
        if rs != 1 or cs != 1 or len(text) < 3 or c + 1 >= len(item.col_edges) or r + 1 >= len(item.row_edges):
            continue
        x0, x1 = item.col_edges[c] / sx, item.col_edges[c + 1] / sx
        y0, y1 = item.row_edges[r] / sy, item.row_edges[r + 1] / sy
        inset = 0.08 * (y1 - y0)
        m = measure_block(rgb, gray, (int(x0 + inset), int(y0 + inset), int(x1 - inset), int(y1 - inset)))
        if len(m.lines) == 1:
            ln = m.lines[0]
            sizes.append((fit_single_line(text, (ln.x1 - ln.x0) * sx, bold=(r == 0), italic=False), len(text), r))
            gap_l, gap_r = ln.x0 - x0, x1 - ln.x1
            free = max(1.0, gap_l + gap_r)
            item.cell_align[(r, c)] = ("left" if gap_l < 0.25 * free else
                                       "right" if gap_r < 0.25 * free else "center")
    # a column is aligned one way: take the majority of its body cells (header row may differ)
    for col in {c for (_r, c) in item.cell_align}:
        votes = [a for (r, c), a in item.cell_align.items() if c == col and r > 0]
        if votes:
            majority = max(set(votes), key=votes.count)
            item.col_align[col] = majority
            for key in [k for k in item.cell_align if k[1] == col and k[0] > 0]:
                item.cell_align[key] = majority
    header = _weighted_size([(s, n) for s, n, r in sizes if r == 0])
    body = _weighted_size([(s, n) for s, n, r in sizes if r > 0])
    item.header_size = header
    return body or header


def _weighted_size(sizes: list[tuple[float, int]]) -> float | None:
    """Remnants of faint rules inflate short cells: drop outliers, then a median weighted by text length
    (long cells are measured most precisely)."""
    if not sizes:
        return None
    mid = float(np.median([s for s, _ in sizes]))
    kept = sorted([(s, n) for s, n in sizes if 0.75 * mid <= s <= 1.33 * mid] or sizes)
    half, acc = sum(n for _, n in kept) / 2, 0
    for s, n in kept:
        acc += n
        if acc >= half:
            return s
    return kept[-1][0]


def _decorative_rules(gray, rgb, tables, text_boxes, sx, sy, assets: Path) -> list[FloatImage]:
    """Short horizontal rules outside tables (under the agency name, the motto, the subject line):
    kept as transparent crops at their exact position. A stroke lying inside a measured text line
    (long dash, pen stroke through letters) is not a rule."""
    h, w = gray.shape
    mask = line_mask(gray)
    rules = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 25, 1)))
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(rules, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, cw, ch, _area = (int(v) for v in stats[i])
        if cw < w // 25 or ch > h // 150:
            continue
        cx, cy = (x + cw / 2) * sx, (y + ch / 2) * sy
        if any(t.x0 - 6 <= cx <= t.x1 + 6 and t.y0 - 6 <= cy <= t.y1 + 6 for t in tables):
            continue
        mid_y = y + ch / 2
        if any(ty0 < mid_y < ty1 and min(x + cw, tx1) - max(x, tx0) > 0.3 * cw
               for tx0, ty0, tx1, ty1 in text_boxes):
            continue
        # a decorative rule sits just under a printed line it overlaps (agency name, motto, subject);
        # strokes elsewhere are paper texture, folds or plastic sleeves
        if not any(0 <= y - ty1 <= 1.5 * (ty1 - ty0) and min(x + cw, tx1) - max(x, tx0) > 0.5 * cw
                   for tx0, ty0, tx1, ty1 in text_boxes):
            continue
        # draw a clean rule from the rule pixels only: a crop would carry fragments of the letters
        # touching it (descenders, dots under ộ / ạ) onto the re-typeset text
        piece = (_lab[y:y + ch, x:x + cw] == i)
        color = np.median(rgb[y:y + ch, x:x + cw][piece], axis=0).astype(np.uint8)
        rgba = np.zeros((ch, cw, 4), np.uint8)
        rgba[..., :3] = color
        rgba[..., 3] = piece * 255
        png = assets / f"rule_{i}.png"
        cv2.imwrite(str(png), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        out.append(FloatImage(x * sx, y * sy, (x + cw) * sx, (y + ch) * sy, png))
    return out


def _table_size(item: TableItem, body: float, cell_pad: float = 2.0) -> float:
    """Largest size (≤ body) at which single-row cells fit their column on one line, as in the scan."""
    from lxml import html as lh
    from .measure import snap_size, text_width_pt
    rows = lh.fromstring(item.html).xpath("//tr")
    nrows, ncols = _grid_shape(item.html)
    edges = item.col_edges or [item.x0 + (item.x1 - item.x0) * k / ncols for k in range(ncols + 1)]
    fits, taken = [], set()
    for r, tr in enumerate(rows):
        c = 0
        for cell in tr.xpath("./td|./th"):
            while (r, c) in taken:
                c += 1
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, c + dc))
            text = cell.text_content().strip()
            unit = text_width_pt(text, 1.0, bold=(r == 0))
            if rs == 1 and unit > 0 and c + cs < len(edges):
                fits.append((edges[c + cs] - edges[c] - cell_pad) / unit)
            c += cs
    limit = min(fits) if fits else body
    return snap_size(min(body, limit))


# ---------------------------------------------------------------- XY-cut

@dataclass
class VStack:
    children: list
    x0: float
    x1: float


@dataclass
class HSplit:
    columns: list                   # VStack per column
    x0: float
    x1: float


def _bounds(item) -> tuple[float, float, float, float]:
    ink = getattr(item, "ink", None)
    x0, x1 = ink if ink else (item.x0, item.x1)
    return x0, item.y0, x1, item.y1


def _groups(items: list, axis: int, min_gap: float) -> list[list]:
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    items = sorted(items, key=lambda it: _bounds(it)[lo])
    groups, end = [], None
    for it in items:
        b = _bounds(it)
        if groups and b[lo] < end + min_gap:
            groups[-1].append(it)
            end = max(end, b[hi])
        else:
            groups.append([it])
            end = b[hi]
    return groups


def xy_cut(items: list, x0: float, x1: float):
    """Recursive XY-cut: horizontal bands first, then columns inside a band."""
    if len(items) <= 1:
        return VStack(items, x0, x1)
    bands = _groups(items, axis=1, min_gap=0.0)
    if len(bands) > 1:
        children = []
        for band in bands:
            node = xy_cut(band, x0, x1)
            children += node.children if isinstance(node, VStack) else [node]
        return VStack(children, x0, x1)
    cols = _groups(items, axis=0, min_gap=4.0)
    if len(cols) == 1:
        # side-by-side handwritten columns touch their dividing line, so their ink overlaps a little; blocks
        # sharing a band cannot be stacked, so a small overlap (≤ 5 % of the narrower block) is still a column
        narrow = min(_bounds(it)[2] - _bounds(it)[0] for it in items)
        cols = _groups(items, axis=0, min_gap=-0.05 * narrow)
    if len(cols) > 1:
        edges = [x0] + [(max(_bounds(i)[2] for i in a) + min(_bounds(i)[0] for i in b)) / 2
                        for a, b in zip(cols, cols[1:])] + [x1]
        return HSplit([xy_cut(c, edges[k], edges[k + 1]) for k, c in enumerate(cols)], x0, x1)
    # a tall formula's line height (1.25 × its rows) reaches into the row below: a row of side-by-side
    # matrices then joins the formula's band and, with no column gap past the wide formula, gets stacked.
    # Bands overlapping by ≤ 30 % of the shorter item are still separate rows
    short = min(_bounds(it)[3] - _bounds(it)[1] for it in items)
    rows = _groups(items, axis=1, min_gap=-0.3 * short)
    if len(rows) > 1:
        children = []
        for row in rows:
            node = xy_cut(row, x0, x1)
            children += node.children if isinstance(node, VStack) else [node]
        return VStack(children, x0, x1)
    return VStack(sorted(items, key=lambda it: _bounds(it)[1]), x0, x1)


def node_top(node) -> float:
    if isinstance(node, (Para, TableItem)):
        return node.y0
    kids = node.children if isinstance(node, VStack) else node.columns
    return min(node_top(k) for k in kids) if kids else 0.0


def node_bottom(node) -> float:
    if isinstance(node, (Para, TableItem)):
        return node.y1
    kids = node.children if isinstance(node, VStack) else node.columns
    return max(node_bottom(k) for k in kids) if kids else 0.0


def build_from_middle(middle: dict, image_path: Path, assets: Path,
                      page_pt: tuple[float, float] | None = None) -> list[Page]:
    return [build_page(p, image_path, assets / f"p{i}", page_pt) for i, p in enumerate(middle.get("pages", []))]


def decode_image_base64(data: str, dst: Path) -> Path:
    dst.write_bytes(base64.b64decode(data.split(",", 1)[-1]))
    return dst
