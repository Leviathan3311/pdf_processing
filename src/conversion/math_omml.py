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


@lru_cache(maxsize=512)
def _mathml(latex: str) -> etree._Element | None:
    try:
        from latex2mathml.converter import convert
        return etree.fromstring(convert(latex))
    except Exception:
        return None


def display_text(latex: str) -> str:
    """What the formula shows (ℝ, →, x2 …): used to estimate its width when fitting font sizes."""
    mml = _mathml(latex)
    if mml is None:
        return latex
    return "".join(t.strip() for t in mml.itertext())


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
        if tag in ("msup", "msub") and len(kids) >= 2:
            return walk(kids[0]) + 0.7 * walk(kids[1])
        if tag == "msubsup" and len(kids) >= 3:
            return walk(kids[0]) + 0.7 * max(walk(kids[1]), walk(kids[2]))
        if tag in ("msqrt", "mroot"):
            return 0.8 + sum(walk(k) for k in kids[:1])
        if tag in ("munder", "mover", "munderover") and kids:
            return max(walk(k) for k in kids)
        return sum(walk(k) for k in kids)

    mml = _mathml(latex)
    return walk(mml) if mml is not None else leaf(latex)


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
