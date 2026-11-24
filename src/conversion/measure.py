"""Measure typography of an OCR block from the page image (MinerU gives no font, size, style or colour).

Everything is measured, not guessed:
  - text lines: horizontal ink projection inside the block (diacritic bands merged into their line)
  - font size: real glyph advance widths (Times New Roman / Arial TTF via Pillow) fitted to measured line widths
  - bold: stroke density relative to the page's regular text; italic: column-gap sharpness after shearing
  - colour: median ink colour; strongly coloured ink (blue pen, red seal) is reported as such
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

FONT_DIR = Path("C:/Windows/Fonts")
FONT_FILES = {
    ("Times New Roman", False, False): "times.ttf",
    ("Times New Roman", True, False): "timesbd.ttf",
    ("Times New Roman", False, True): "timesi.ttf",
    ("Times New Roman", True, True): "timesbi.ttf",
    ("Arial", False, False): "arial.ttf",
    ("Arial", True, False): "arialbd.ttf",
    ("Arial", False, True): "ariali.ttf",
    ("Arial", True, True): "arialbi.ttf",
}
DEFAULT_FONT = "Times New Roman"
COMPONENT_COLOUR_MIN = 10.0   # hue_offset of a stroke that is pen ink, not print (see _ink_color for blocks)


@lru_cache(maxsize=None)
def _font(name: str, bold: bool, italic: bool) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / FONT_FILES[(name, bold, italic)]), size=1000)


def text_width_pt(text: str, size: float, name: str = DEFAULT_FONT, bold: bool = False, italic: bool = False) -> float:
    """Advance width of `text` set at `size` pt."""
    return _font(name, bold, italic).getlength(text) / 1000.0 * size


@lru_cache(maxsize=1)
def _math_font():
    """Cambria Math: the font Word draws equations with (face 1 of cambria.ttc)."""
    try:
        return ImageFont.truetype(str(FONT_DIR / "cambria.ttc"), size=1000, index=1)
    except OSError:
        return _font(DEFAULT_FONT, False, True)


def wrap(text: str, width_pt: float, size: float, name: str, bold: bool, italic: bool,
         first_indent: float = 0.0) -> list[str]:
    """Greedy word wrap, the way Word breaks a left/justified paragraph (first line narrower by the indent)."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        limit = width_pt - (first_indent if not lines else 0.0)
        if cur and text_width_pt(trial, size, name, bold, italic) > limit:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


@dataclass
class Line:
    y0: float          # px, in the page image
    y1: float
    x0: float
    x1: float
    density: float     # ink pixels / line box area
    darkness: float = 255.0   # mean grey of the darkest third of ink pixels: thick (bold) strokes reach deeper black
    gaps: list = field(default_factory=list)   # wide blank runs inside the line (px, start, end): column gaps
    blanks: list = field(default_factory=list)  # every blank run wider than a quarter letter height (word gaps too)
    letter_h: float = 0.0                       # typical letter height (px): stacked diacritics do not inflate it


@dataclass
class Typo:
    size: float = 13.0
    font: str = DEFAULT_FONT
    bold: bool = False
    italic: bool = False
    color: str | None = None       # hex RGB, None = black


@dataclass
class BlockMeasure:
    lines: list[Line] = field(default_factory=list)
    colored_ink: bool = False      # mostly non-black ink: handwriting, signature, seal
    color: str | None = None
    italic: bool = False


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary ink mask (uint8 0/1) robust to uneven paper: Otsu capped at a dark threshold."""
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (gray < min(otsu, 170)).astype(np.uint8)


def _remove_rules(mask: np.ndarray) -> np.ndarray:
    """Drop long horizontal / vertical strokes (table lines, underlines) so they do not count as text."""
    h, w = mask.shape
    out = mask.copy()
    if w > 20:
        horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 6), 1)))
        out[horiz > 0] = 0
    if h > 20:
        vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 2))))
        out[vert > 0] = 0
    return out


def _runs(flags) -> list[list[int]]:
    runs, start = [], None
    for i, v in enumerate(list(flags) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    return runs


def text_lines(mask: np.ndarray, x_off: float = 0, y_off: float = 0, ref_height: float | None = None,
               colored: np.ndarray | None = None, has_math: bool = False) -> list[Line]:
    """Text lines from connected components.

    Letter-sized components define lines (grouped by their vertical centre); small components (diacritics,
    dots, commas) never start a line — Vietnamese marks often touch the neighbouring line, which breaks
    projection-based line finding.
    """
    mask = _remove_rules(mask)
    h, w = mask.shape
    if h == 0 or w == 0 or mask.max() == 0:
        return []
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [tuple(int(v) for v in stats[i][:5]) for i in range(1, n) if stats[i][4] >= 3]   # x, y, w, h, area
    if not comps:
        return []
    letter_h = float(np.percentile([c[3] for c in comps], 75))
    # a stroke far wider than any letter and thin is a rule (a short underline survives _remove_rules)
    comps = [c for c in comps if not (c[2] > 4 * letter_h and c[3] < 0.35 * letter_h)] or comps
    main = sorted((c for c in comps if c[3] >= 0.45 * letter_h), key=lambda c: c[1] + c[3] / 2)
    groups: list[list[tuple]] = []
    for c in main:
        centre = c[1] + c[3] / 2
        if groups:
            g = groups[-1]
            g_centre = float(np.median([d[1] + d[3] / 2 for d in g]))
            if abs(centre - g_centre) < 0.5 * letter_h:
                g.append(c)
                continue
        groups.append([c])

    def extent(g):
        return min(c[1] for c in g), max(c[1] + c[3] for c in g)

    # real lines never overlap much vertically; a line with a tall formula (stacked fraction, scripts)
    # splits into overlapping groups. On blurry scans neighbouring text lines touch a little, so the
    # stricter merge applies only to blocks that contain a formula.
    overlap_limit = 0.15 if has_math else 0.4
    merged = True
    while merged and len(groups) > 1:
        merged = False
        groups.sort(key=lambda g: extent(g)[0])
        for i in range(len(groups) - 1):
            (a0, a1), (b0, b1) = extent(groups[i]), extent(groups[i + 1])
            if min(a1, b1) - max(a0, b0) > overlap_limit * min(a1 - a0, b1 - b0):
                groups[i] += groups.pop(i + 1)
                merged = True
                break

    while len(groups) > 1:                             # stray marks between lines: fold into the nearest line
        heights = [extent(g)[1] - extent(g)[0] for g in groups]
        counts = [len(g) for g in groups]
        ref = max(float(np.median(heights)), 0.0 if ref_height is None else ref_height)
        ref_count = float(np.median(counts))
        spans = [extent(g) for g in groups]

        def gap_to_neighbour(i: int) -> float:
            others = [k for k in range(len(groups)) if k != i]
            return min(max(spans[k][0] - spans[i][1], spans[i][0] - spans[k][1], 0) for k in others)

        # fold only marks hugging another line (stray accents, a fraction's numerator / denominator);
        # a short real line ("A ∪ C; A ∩ C.") sits a normal line gap away and stays
        small = [i for i, hgt in enumerate(heights)
                 if (hgt < 0.55 * ref or (counts[i] < 0.3 * ref_count and hgt < 0.8 * ref))
                 and gap_to_neighbour(i) < 0.35 * ref]
        if not small:
            break
        i = min(small, key=lambda k: heights[k])
        centre = sum(extent(groups[i])) / 2
        others = [k for k in range(len(groups)) if k != i]
        j = min(others, key=lambda k: abs(sum(extent(groups[k])) / 2 - centre))
        stray = groups.pop(i)
        groups[j - 1 if j > i else j] += stray
        groups.sort(key=lambda g: extent(g)[0])

    def is_colored(c) -> bool:
        if colored is None:
            return False
        x, y, cw, ch, _ = c
        values = colored[y:y + ch, x:x + cw][mask[y:y + ch, x:x + cw] > 0]
        return len(values) > 0 and float(np.median(values)) >= COMPONENT_COLOUR_MIN

    lines = []
    for g in groups:
        y0 = min(c[1] for c in g)
        y1 = max(c[1] + c[3] for c in g)
        # Horizontal extent uses every component in the line band — a leading dash or a final ";" is
        # small but is part of the line's width — and only printed (black) ink: a trailing pen scribble
        # is not text, while coloured ink *between* printed letters (a hand-filled number) stays inside.
        small = [c for c in comps if c[3] < 0.45 * letter_h and y0 <= c[1] + c[3] / 2 <= y1]
        printed = [c for c in g + small if not is_colored(c)] or g
        x0 = min(c[0] for c in printed)
        x1 = max(c[0] + c[2] for c in printed)
        area = max(1, (y1 - y0) * (x1 - x0))
        # column gaps: blank runs much wider than a word space (exercise grids "a) …      b) …")
        occupied = np.zeros(x1 - x0 + 1, bool)
        for c in g + small:                            # any ink fills a gap, pen too ("Số: 133 /…")
            occupied[max(0, c[0] - x0):max(0, c[0] + c[2] - x0)] = True
        line_letter_h = float(np.median([c[3] for c in g]))
        blanks = [(s + x0 + x_off, e + x0 + x_off) for s, e in _runs(~occupied) if e - s > 0.25 * line_letter_h]
        gaps = [(s, e) for s, e in blanks if e - s > 1.5 * (y1 - y0)]
        lines.append(Line(y0 + y_off, y1 + y_off, x0 + x_off, x1 + x_off, float(mask[y0:y1, x0:x1].sum()) / area,
                          gaps=gaps, blanks=blanks, letter_h=line_letter_h))
    return lines


def _italic_score(mask: np.ndarray) -> float:
    """>1 when columns separate better after a ~12° shear, i.e. the text leans like italics."""
    h, w = mask.shape
    if h < 8 or w < 20:
        return 1.0
    def sharpness(shear: float) -> float:
        m = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
        sheared = cv2.warpAffine(mask.astype(np.float32), m, (w, h))
        proj = sheared.sum(axis=0)
        return float((proj == 0).sum()) / w      # share of empty columns: gaps between letters
    base = sharpness(0.0)
    return sharpness(0.21) / base if base > 0 else 1.0


def hue_offset(rgb: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel colour distance from the page's own print colour, brightness removed.

    A pixel's colour vector is rgb − mean(rgb) (grey = 0). Comparing it with the page's typical ink vector
    `ref` makes "coloured" relative: a whole letter printed with a purple cast is not pen ink, a blue
    signature or red seal on that page still is."""
    rgb = rgb.astype(np.float32)
    vec = rgb - rgb.mean(axis=-1, keepdims=True)
    if ref is not None:
        vec = vec - ref
    return np.linalg.norm(vec, axis=-1)


def ink_reference(rgb: np.ndarray, gray: np.ndarray,
                  boxes: list[tuple[int, int, int, int]] | None = None) -> np.ndarray:
    """The page's typical print colour vector (median over ink pixels).

    `boxes`: the text regions (px). Only their ink counts, so a phone photo's table top or background around
    the sheet (dark, brown) does not become the "print colour" and turn every letter into "coloured ink"."""
    step = max(1, max(gray.shape) // 1200)
    if boxes:
        parts = []
        for x0, y0, x1, y1 in boxes:
            g = gray[y0:y1:step, x0:x1:step]
            if g.size:
                parts.append(rgb[y0:y1:step, x0:x1:step].astype(np.float32)[ink_mask(g).astype(bool)])
        pixels = np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)
        if len(pixels) >= 100:
            return np.median(pixels - pixels.mean(axis=1, keepdims=True), axis=0)
    small_rgb, small_gray = rgb[::step, ::step].astype(np.float32), gray[::step, ::step]
    pixels = small_rgb[ink_mask(small_gray).astype(bool)]
    if len(pixels) < 100:
        return np.zeros(3, np.float32)
    return np.median(pixels - pixels.mean(axis=1, keepdims=True), axis=0)


# |print colour vector| (rgb − grey): print measured ≤ 6.4 on 12 pages (scans, screenshots, photos with a cast);
# pages written entirely by pen (blue ink) 13.1–14.5
PEN_PAGE_CHROMA = 10.0


def is_pen_page(ink_ref: np.ndarray) -> bool:
    """The page's own main ink is pen colour: the page is handwritten, colour no longer marks an annotation."""
    return float(np.linalg.norm(ink_ref)) >= PEN_PAGE_CHROMA


def _ink_color(rgb: np.ndarray, mask: np.ndarray, ref: np.ndarray | None = None) -> tuple[str | None, bool]:
    pixels = rgb[mask.astype(bool)]
    if len(pixels) < 20:
        return None, False
    r, g, b = (int(np.median(pixels[:, i])) for i in range(3))
    # measured on 4 pages (after flattening): print ≤ 7.4, print under a seal 13.9, pen / seal / signature 22–59
    if float(np.median(hue_offset(pixels, ref))) >= 16:
        return f"{r:02X}{g:02X}{b:02X}", True
    return None, False                                    # the page's own print colour


def measure_block(rgb: np.ndarray, gray: np.ndarray, box_px: tuple[int, int, int, int],
                  ref_height: float | None = None, has_math: bool = False,
                  ink_ref: np.ndarray | None = None, pen_page: bool = False) -> BlockMeasure:
    """`ref_height`: typical text line height on the page (px), used to fold stray mark groups.
    `ink_ref`: the page's print colour (`ink_reference`), so colour is judged relative to the page.
    `pen_page`: the whole page is handwritten — pen strokes at a line's ends are text, not a scribble."""
    x0, y0, x1, y1 = box_px
    pad = 2
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(gray.shape[1], x1 + pad), min(gray.shape[0], y1 + pad)
    crop = gray[y0:y1, x0:x1]
    mask = ink_mask(crop)
    color, colored = _ink_color(rgb[y0:y1, x0:x1], mask, ink_ref)
    chroma = hue_offset(rgb[y0:y1, x0:x1], ink_ref)              # per-pixel colour distance from print
    lines = text_lines(mask, x0, y0, ref_height, colored=None if pen_page else chroma, has_math=has_math)
    for line in lines:
        g = gray[int(line.y0):int(line.y1), int(line.x0):int(line.x1)]
        ink = np.sort(g[_remove_rules(ink_mask(g)).astype(bool)])
        if len(ink):
            line.darkness = float(ink[:max(1, len(ink) // 3)].mean())
    return BlockMeasure(lines=lines, colored_ink=colored, color=color,
                        italic=_italic_score(_remove_rules(mask)) > 1.1)   # measured: italic 1.21–1.31, upright ≤ 0.95


def _strip_marks(text: str) -> str:
    """Base letters only: Vietnamese marks are separate ink components and are not in the line height."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def unit_base_height(text: str, name: str, bold: bool, italic: bool) -> float:
    """Ink height (pt per 1 pt of font size) of the base letters of `text`."""
    box = _font(name, bold, italic).getbbox(_strip_marks(text) or "x")
    return (box[3] - box[1]) / 1000.0


def is_bold(darkness: float, page_median_darkness: float) -> bool | None:
    """Bold if the stroke core is clearly darker than the page's typical (regular) text.

    Measured on a 565 px scan: bold 18–45, regular ≥ 47, median ~56 → bold below 0.83 × median and at least
    8 grey levels darker. On a crisp page (median < 25, every stroke already black) darkness cannot tell:
    returns None so the caller falls back to the block type. Stroke width and ink coverage were tried and
    do not separate bold from regular on either kind of page."""
    if page_median_darkness < 25:
        return None
    return darkness < 0.83 * page_median_darkness and page_median_darkness - darkness >= 8


def cluster_sizes(sizes: list[float], tol: float = 0.8) -> list[float]:
    """Pull noisy per-line sizes to shared values (documents use few sizes); round to 0.5 pt."""
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    out = list(sizes)
    group: list[int] = []
    for i in order + [None]:
        if i is not None and (not group or sizes[i] - sizes[group[0]] <= tol):
            group.append(i)
            continue
        if group:
            value = snap_size(float(np.median([sizes[g] for g in group])))
            for g in group:
                out[g] = value
        group = [i] if i is not None else []
    return out


def fit_single_line(text: str, width_pt: float, bold: bool, italic: bool, name: str = DEFAULT_FONT) -> float:
    unit = text_width_pt(text, 1.0, name, bold, italic)
    return width_pt / unit if unit > 0 else 0.0


def split_by_widths(text: str, widths: list[float]) -> list[str]:
    """Distribute the words of `text` over lines whose measured ink widths are `widths` (any unit)."""
    words = text.split()
    if len(widths) <= 1 or len(words) <= 1:
        return [text]
    total_w = sum(widths)
    targets = np.cumsum(widths)[:-1] / total_w          # where each line should end, as a share of the text
    unit = [text_width_pt(w + " ", 1.0) for w in words]
    cum = np.cumsum(unit) / sum(unit)
    cuts, start = [], 0
    for t in targets:
        k = int(np.argmin(np.abs(cum - t)))             # word index after which the line breaks
        k = max(k, start)
        cuts.append(k + 1)
        start = k + 1
    out, prev = [], 0
    for c in cuts + [len(words)]:
        out.append(" ".join(words[prev:c]))
        prev = c
    return [o for o in out if o] or [text]


def fit_paragraph(text: str, width_pt: float, n_lines: int, last_width_pt: float,
                  bold: bool, italic: bool, name: str = DEFAULT_FONT,
                  pitch_pt: float | None = None, first_indent: float = 0.0,
                  line_widths: list[float] | None = None) -> float:
    """Size whose greedy wrap at `width_pt` gives `n_lines` lines whose widths match the measured ones.
    With a measured line pitch the search stays within plausible line spacing (1.0–1.7 × size).

    Every measured line width counts (`line_widths`; else only the last line): several sizes give the
    right line count, and the last line alone can match at a wrong size by coincidence."""
    lo, hi = (pitch_pt / 1.7, pitch_pt / 1.0) if pitch_pt else (7.0, 28.0)
    best, best_err = 0.0, float("inf")
    for size in np.arange(max(6.0, lo), min(36.0, hi) + 0.01, 0.25):
        lines = wrap(text, width_pt, float(size), name, bold, italic, first_indent)
        err = abs(len(lines) - n_lines) * 1000
        if lines and line_widths and len(lines) == len(line_widths):
            err += float(np.mean([abs(text_width_pt(l, float(size), name, bold, italic) - w)
                                  for l, w in zip(lines, line_widths)]))
        elif lines:
            err += abs(text_width_pt(lines[-1], float(size), name, bold, italic) - last_width_pt)
        if err < best_err:
            best, best_err = float(size), err
    return best


def snap_size(size: float) -> float:
    return max(6.0, min(36.0, round(size * 2) / 2))
