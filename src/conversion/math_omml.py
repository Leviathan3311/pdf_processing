"""LaTeX (MinerU equations) → native, editable Word equations (OMML).

LaTeX → MathML with latex2mathml, MathML → OMML with Microsoft Office's own MML2OMML.XSL (the stylesheet
Word itself uses), so the result is what typing the equation in Word would give. If the stylesheet is
missing or a formula does not convert, callers fall back to plain text (never drop the content).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from lxml import etree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XSL_CANDIDATES = [
    Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL"),
]


@lru_cache(maxsize=1)
def _xslt():
    for path in XSL_CANDIDATES:
        if path.exists():
            return etree.XSLT(etree.parse(str(path)))
    return None


_FENCES = {"[": "]", "(": ")", "{": "}", "|": "|", "‖": "‖", "⌊": "⌋", "⌈": "⌉", "⟨": "⟩"}


def _fence_matrices(root: etree._Element) -> None:
    """`[ <mtable> ]` → `<mfenced open="[" close="]">`: Office's MML2OMML turns only mfenced into a Word
    delimiter that grows with its content; loose brackets stay one text row tall beside a 3-row matrix."""
    ns = etree.QName(root).namespace
    tag = (lambda t: f"{{{ns}}}{t}") if ns else (lambda t: t)
    def name(e) -> str:
        return etree.QName(e).localname

    for row in list(root.iter(tag("mrow"), tag("math"))):
        kids = list(row)
        for i in range(len(kids) - 2):                    # bracket, matrix, bracket anywhere in the row
            a, mid, b = kids[i:i + 3]
            if (name(a) == "mo" and name(mid) == "mtable" and name(b) == "mo"
                    and _FENCES.get((a.text or "").strip()) == (b.text or "").strip()):
                fenced = etree.Element(tag("mfenced"), open=a.text.strip(), close=b.text.strip(), separators="")
                row.replace(a, fenced)
                fenced.append(mid)                        # moves the matrix into the fence
                row.remove(b)


@lru_cache(maxsize=512)
def _mathml(latex: str) -> etree._Element | None:
    try:
        from latex2mathml.converter import convert
        root = etree.fromstring(convert(latex))
    except Exception:
        return None
    _fence_matrices(root)
    return root


def display_text(latex: str) -> str:
    """What the formula shows (ℝ, →, x2 …): used to estimate its width when fitting font sizes."""
    mml = _mathml(latex)
    if mml is None:
        return latex
    return "".join(t.strip() for t in mml.itertext())


_LARGE_OPERATORS = set("∑∏∐∫∬∭∮⋃⋂⋁⋀")
_SPACED_OPERATORS = set("=<>≤≥≠≈+−-→↦∈∉⊂⊆∪∩×÷±∓∖")


def width_em(latex: str) -> float:
    """Width of the typeset formula in em, following its structure (a fraction is as wide as its wider part,
    scripts are smaller) — the linear text "2x1+x2" would overstate it badly."""
    from .measure import _math_font
    font = _math_font()

    def leaf(text: str) -> float:
        return font.getlength(text) / 1000.0

    def walk(node) -> float:
        tag = etree.QName(node).localname
        kids = list(node)
        if tag in ("mi", "mn", "mtext", "ms"):
            return leaf((node.text or "").strip())
        if tag == "mo":
            text = (node.text or "").strip()
            return leaf(text) + (0.27 if text and text in _SPACED_OPERATORS else 0.0)
        if tag == "mfrac" and len(kids) >= 2:
            return max(walk(kids[0]), walk(kids[1])) * 0.9 + 0.2
        if tag in ("msub", "msup", "msubsup", "munder", "mover", "munderover") and len(kids) >= 2                 and (kids[0].text or "").strip() in _LARGE_OPERATORS:
            # Word sets ∑ / ∫ as an n-ary operator: a full-size glyph with its limits and a gap after it
            return 1.1 + 0.7 * max(walk(k) for k in kids[1:]) + 0.4
        if tag in ("msup", "msub") and len(kids) >= 2:
            return walk(kids[0]) + 0.7 * walk(kids[1])
        if tag == "msubsup" and len(kids) >= 3:
            return walk(kids[0]) + 0.7 * max(walk(kids[1]), walk(kids[2]))
        if tag in ("msqrt", "mroot"):
            return 0.8 + sum(walk(k) for k in kids[:1])
        if tag in ("munder", "mover", "munderover") and kids:
            return max(walk(k) for k in kids)
        if tag == "mtable":                              # matrix: widest cell per column, not every cell
            rows = [[c for c in r if etree.QName(c).localname == "mtd"] for r in kids]
            ncols = max((len(r) for r in rows), default=0)
            cols = [max((walk(r[j]) for r in rows if j < len(r)), default=0.0) for j in range(ncols)]
            return sum(cols) + 1.0 * max(0, ncols - 1)   # Word's column gap, measured
        if tag == "mfenced":                             # stretched brackets grow wider with the rows inside
            rows = sum(1 for _ in node.iter("{*}mtr")) or 1
            return sum(walk(k) for k in kids) + 0.3 * rows + 0.3
        return sum(walk(k) for k in kids)

    mml = _mathml(latex)
    return walk(mml) if mml is not None else leaf(latex)


def height_em(latex: str) -> float:
    """Height of the typeset formula in em: 1 for a plain line, more for fractions, matrices, big operators
    with limits. An exact line height measured from one text row would clip a 3-row matrix."""
    def walk(node) -> float:
        tag = etree.QName(node).localname
        kids = list(node)
        if tag in ("mi", "mn", "mtext", "ms", "mo"):
            return 1.0
        if tag == "mfrac" and len(kids) >= 2:
            return 0.85 * (walk(kids[0]) + walk(kids[1])) + 0.15
        if tag in ("msup", "msub", "msubsup") and kids:
            return walk(kids[0]) + 0.3
        if tag in ("munder", "mover") and len(kids) >= 2:
            accent = etree.QName(kids[1]).localname == "mo"         # hat, bar, underline
            return walk(kids[0]) + (0.25 if accent else 0.6 * walk(kids[1]))
        if tag == "munderover" and len(kids) >= 3:
            return walk(kids[0]) + 0.6 * (walk(kids[1]) + walk(kids[2]))
        if tag == "mtable":
            return sum(max((walk(c) for c in r), default=1.0) for r in kids) + 0.25 * max(0, len(kids) - 1)
        return max((walk(k) for k in kids), default=1.0)

    mml = _mathml(latex)
    return walk(mml) if mml is not None else 1.0


def latex_to_omml(latex: str, size_pt: float | None = None) -> etree._Element | None:
    """<m:oMath> element for `latex`, with every math run set to `size_pt`; None if it cannot convert."""
    xslt, mml = _xslt(), _mathml(latex)
    if xslt is None or mml is None:
        return None
    try:
        omml = xslt(mml).getroot()
    except Exception:
        return None
    if omml is None or etree.QName(omml).localname != "oMath":
        return None
    if size_pt:
        half_points = str(int(round(size_pt * 2)))
        for run in omml.iter(f"{{{M_NS}}}r"):
            rpr = run.find(f"{{{W_NS}}}rPr")
            if rpr is None:
                rpr = etree.SubElement(run, f"{{{W_NS}}}rPr")
                run.insert(0, rpr)
            fonts = etree.SubElement(rpr, f"{{{W_NS}}}rFonts")
            for attr in ("ascii", "hAnsi", "cs"):
                fonts.set(f"{{{W_NS}}}{attr}", "Cambria Math")
            for tag in ("sz", "szCs"):
                etree.SubElement(rpr, f"{{{W_NS}}}{tag}").set(f"{{{W_NS}}}val", half_points)
    return omml
