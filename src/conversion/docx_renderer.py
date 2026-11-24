"""Write layout Pages (conversion.layout) to an editable DOCX that looks like the scan (CLAUDE.md section 7).

- Text stays in the body flow (never in text boxes): indents, exact line heights and space-before reproduce
  measured positions; side-by-side regions become borderless tables with fixed column widths.
- Real tables are Word tables with merged cells, measured column widths and row heights.
- Coloured ink (signatures, handwriting) and seals are transparent PNGs anchored at their exact position.
"""
from __future__ import annotations

import copy
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Emu, Pt, RGBColor
from lxml import html as lh

from .layout import HSplit, Page, Para, TableItem, VStack, node_bottom, node_top, xy_cut

ALIGN = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
         "right": WD_ALIGN_PARAGRAPH.RIGHT, "justify": WD_ALIGN_PARAGRAPH.JUSTIFY}
EMU_PER_PT = 12700


# ---------------------------------------------------------------- low-level helpers

def _set_fonts(run, name: str) -> None:
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)


def _format_para(p, *, left=0.0, right=0.0, first=0.0, before=0.0, line=None, align="left", outline=None):
    pf = p.paragraph_format
    pf.left_indent, pf.right_indent = Pt(max(0.0, left)), Pt(max(0.0, right))
    pf.first_line_indent = Pt(first)
    pf.space_before, pf.space_after = Pt(max(0.0, before)), Pt(0)
    if line:
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        pf.line_spacing = Pt(line)
    pf.alignment = ALIGN[align]
    pf.widow_control = False
    if outline is not None:
        ppr = p._p.get_or_add_pPr()
        lvl = OxmlElement("w:outlineLvl")
        lvl.set(qn("w:val"), str(max(0, int(outline) - 1)))
        ppr.append(lvl)


def _add_run(p, text, typo):
    run = p.add_run(text)
    _set_fonts(run, typo.font)
    run.font.size = Pt(typo.size)
    run.font.bold = typo.bold
    run.font.italic = typo.italic
    if typo.color:
        run.font.color.rgb = RGBColor.from_string(typo.color)
    return run


def _line_width(runs) -> float:
    """Typeset width (pt) of one line of runs: Times New Roman text, Cambria Math formulas."""
    from .math_omml import width_em
    from .measure import text_width_pt
    total = 0.0
    for r in runs:
        if r.latex:
            total += width_em(r.latex) * r.typo.size
        else:
            total += text_width_pt(r.text, r.typo.size, r.typo.font, r.typo.bold, r.typo.italic)
    return total


def _add_math(p, run) -> None:
    """Native Word equation (editable); falls back to the formula's displayed text if it cannot convert."""
    from .math_omml import latex_to_omml
    omml = latex_to_omml(run.latex, run.typo.size)
    if omml is None:
        _add_run(p, run.text, run.typo)
    else:
        p._p.append(omml)


def _no_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tbl_pr.append(borders)


def _zero_cell_margins(table) -> None:
    mar = OxmlElement("w:tblCellMar")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), "0")
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    table._tbl.tblPr.append(mar)


def _fixed_widths(table, widths_pt: list[float]) -> None:
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    tw = OxmlElement("w:tblW")
    tw.set(qn("w:w"), str(int(sum(widths_pt) * 20)))
    tw.set(qn("w:type"), "dxa")
    tbl_pr.append(tw)
    grid = table._tbl.tblGrid
    for col, w in zip(grid.findall(qn("w:gridCol")), widths_pt):
        col.set(qn("w:w"), str(int(w * 20)))
    for row in table.rows:
        for cell, w in zip(row.cells, widths_pt):
            cell.width = Pt(w)


def _indent_table(table, left_pt: float) -> None:
    ind = OxmlElement("w:tblInd")
    ind.set(qn("w:w"), str(int(left_pt * 20)))
    ind.set(qn("w:type"), "dxa")
    table._tbl.tblPr.append(ind)


def _spacer(container, height: float):
    p = container.add_paragraph()
    _format_para(p, line=max(1.0, height))
    p.paragraph_format.line_spacing = Pt(max(1.0, height))
    run = p.add_run("")
    run.font.size = Pt(1)
    return p


def _anchor_picture(paragraph, png: Path, x_pt: float, y_pt: float, w_pt: float, h_pt: float, pid: int) -> None:
    """Insert `png` as a floating picture positioned relative to the page (no text wrapping)."""
    run = paragraph.add_run()
    run.add_picture(str(png), width=Emu(int(w_pt * EMU_PER_PT)), height=Emu(int(h_pt * EMU_PER_PT)))
    inline = run._r.xpath(".//wp:inline")[0]
    graphic = copy.deepcopy(inline.find(qn("a:graphic")))
    cx, cy = int(w_pt * EMU_PER_PT), int(h_pt * EMU_PER_PT)
    anchor = parse_xml(
        f'<wp:anchor {nsdecls("wp", "a", "pic", "r")} distT="0" distB="0" distL="0" distR="0" simplePos="0" '
        f'relativeHeight="{251658240 + pid}" behindDoc="0" locked="0" layoutInCell="1" allowOverlap="1">'
        f'<wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{int(x_pt * EMU_PER_PT)}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{int(y_pt * EMU_PER_PT)}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:wrapNone/>'
        f'<wp:docPr id="{1000 + pid}" name="Float {pid}"/><wp:cNvGraphicFramePr/></wp:anchor>')
    anchor.append(graphic)
    inline.getparent().replace(inline, anchor)


# ---------------------------------------------------------------- content

def _render_para(container, para: Para, cx0: float, cx1: float, before: float, first_p=None):
    slack = 0.06 * (para.x1 - para.x0) + 4.0
    for i, line in enumerate(para.lines):
        p = first_p if (first_p is not None and i == 0) else container.add_paragraph()
        if para.wrap:
            left, right = para.x0 - cx0, cx1 - para.x1 - 1.0
            align = "justify" if para.justify else "left"
            _format_para(p, left=left, right=right, first=para.first_indent, before=before,
                         line=para.line_height, align=align, outline=para.outline)
        else:
            if para.tab_stops:                             # grid line: left edge + tab stops at measured columns
                lx0 = para.line_x[i][0] if para.line_x else para.x0
                left, right, align = lx0 - cx0, 0.0, "left"
            else:                                          # centred within the line's own measured box:
                lx0, lx1 = para.line_x[i] if para.line_x else (para.x0, para.x1)   # width error splits evenly
                # a standalone line must never wrap: widen the box when the text needs more room than was
                # measured (part of the line hidden under a seal, faint print)
                need = _line_width(line) * 1.04
                line_slack = max(0.06 * (lx1 - lx0) + 4.0, (need - (lx1 - lx0)) / 2 + 2.0)
                left, right, align = lx0 - cx0 - line_slack, cx1 - lx1 - line_slack, "center"
                if left < 0:                               # keep the widened box inside the column
                    right, left = max(0.0, right + left), 0.0
                if right < 0:
                    left, right = max(0.0, left + right), 0.0
            _format_para(p, left=left, right=right, before=before if i == 0 else 0.0,
                         line=para.line_height, align=align, outline=para.outline if i == 0 else None)
            for x in para.tab_stops:                       # Word tab positions count from the column start
                p.paragraph_format.tab_stops.add_tab_stop(Pt(x - cx0))
        for r in line:
            if r.latex:
                _add_math(p, r)
            else:
                _add_run(p, r.text, r.typo)
        before = 0.0
    return p


def _render_table(container, item: TableItem, cx0: float, cx1: float):
    rows = lh.fromstring(item.html).xpath("//tr")
    grid: dict[tuple[int, int], tuple] = {}
    ncols = 0
    for r, tr in enumerate(rows):
        c = 0
        for cell in tr.xpath("./td|./th"):
            while (r, c) in grid:
                c += 1
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = (r, c) if (dr or dc) else (r, c, rs, cs, cell.text_content().strip())
            c += cs
        ncols = max(ncols, c)
    nrows = len(rows)
    table = container.add_table(rows=nrows, cols=ncols)
    if item.borders:
        table.style = "Table Grid"
    mar = OxmlElement("w:tblCellMar")
    for edge in ("left", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), "20")                                   # 1 pt (layout._table_size pads 2 pt)
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    table._tbl.tblPr.append(mar)
    if item.col_edges:
        widths = [b - a for a, b in zip(item.col_edges, item.col_edges[1:])]
        left = item.col_edges[0] - cx0
    else:
        widths = [(item.x1 - item.x0) / ncols] * ncols
        left = item.x0 - cx0
    _fixed_widths(table, widths)
    _indent_table(table, left)
    if item.row_edges:
        for row, (a, b) in zip(table.rows, zip(item.row_edges, item.row_edges[1:])):
            row.height = Pt(b - a)
            row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    for (r, c), v in sorted(grid.items()):
        if len(v) != 5:
            continue
        _, _, rs, cs, text = v
        cell = table.cell(r, c)
        if rs > 1 or cs > 1:
            cell = cell.merge(table.cell(r + rs - 1, c + cs - 1))
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        align = (item.cell_align.get((r, c)) or (item.col_align.get(c) if r > 0 else None)
                 or ("center" if len(text) <= 40 else "left"))
        _format_para(p, left=1.5 if align == "left" else 0.0, align=align)
        from .measure import Typo
        size = item.header_size if (r == 0 and item.header_size) else item.size
        _add_run(p, text, Typo(size=size, bold=(r == 0 and item.header_bold)))
    return table


def _render_node(container, node, cx0: float, cx1: float, top: float, first_p=None):
    """Render a VStack/HSplit into `container` (document body or table cell). `top`: y the container starts at."""
    cursor = top
    pending_after = None                     # last paragraph, to carry a gap before a following table
    kids = node.children if isinstance(node, VStack) else [node]
    for kid in kids:
        gap = node_top(kid) - cursor
        if isinstance(kid, Para):
            pending_after = _render_para(container, kid, cx0, cx1, gap, first_p)
            first_p = None
        else:
            if gap > 0.5:
                if pending_after is not None:
                    pending_after.paragraph_format.space_after = Pt(gap)
                else:
                    _spacer(container, gap) if first_p is None else _format_para(first_p, line=gap)
                    first_p = None
            if isinstance(kid, TableItem):
                _render_table(container, kid, cx0, cx1)
            else:
                _render_split(container, kid, cx0, cx1)
            pending_after = None
        cursor = node_bottom(kid)
    if first_p is not None:                  # empty cell keeps a tiny paragraph
        _format_para(first_p, line=1.0)


def _render_split(container, split: HSplit, cx0: float, cx1: float):
    edges = [col.x0 for col in split.columns] + [split.columns[-1].x1]
    table = container.add_table(rows=1, cols=len(split.columns))
    _no_borders(table)
    _zero_cell_margins(table)
    _fixed_widths(table, [b - a for a, b in zip(edges, edges[1:])])
    _indent_table(table, edges[0] - cx0)
    band_top = node_top(split)
    for cell, col in zip(table.rows[0].cells, split.columns):
        _render_node(cell, col, col.x0, col.x1, band_top, first_p=cell.paragraphs[0])
    return table


# ---------------------------------------------------------------- document

def render_pages(pages: list[Page], out: Path) -> Path:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.space_before = Pt(0)
    pid = 0
    for i, page in enumerate(pages):
        section = doc.sections[0] if i == 0 else doc.add_section()
        section.page_width, section.page_height = Pt(page.width), Pt(page.height)
        section.orientation = WD_ORIENT.LANDSCAPE if page.width > page.height else WD_ORIENT.PORTRAIT
        flow = page.flow
        left = max(14.0, min(it.x0 for it in flow) - 2) if flow else 56.0
        right = max(14.0, page.width - max(it.x1 for it in flow) - 2) if flow else 56.0
        top = max(1.0, min(it.y0 for it in flow) - 1) if flow else 56.0   # content can sit near the edge
        section.left_margin, section.right_margin = Pt(left), Pt(right)
        section.top_margin, section.bottom_margin = Pt(top), Pt(1)    # signatures often sit near the edge
        section.header_distance = section.footer_distance = Pt(0)
        body = doc if i == 0 else doc
        anchor_p = doc.paragraphs[0] if i == 0 and doc.paragraphs else body.add_paragraph()
        _format_para(anchor_p, line=1.0)                  # carries the floating pictures of this page
        for f in page.floats:
            pid += 1
            _anchor_picture(anchor_p, f.png, f.x0, f.y0, f.x1 - f.x0, f.y1 - f.y0, pid)
        root = xy_cut(flow, left, page.width - right)
        if not isinstance(root, VStack):
            root = VStack([root], left, page.width - right)
        _render_node(body, root, left, page.width - right, top + 1.0)
        if i < len(pages) - 1:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.save(str(out))
    return out
