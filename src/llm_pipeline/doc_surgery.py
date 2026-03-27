"""
Document Surgery - Apply modifications directly to .docx XML.

Works directly with the ZIP/XML structure of .docx files,
bypassing python-docx's Document model to preserve ALL formatting,
textboxes, images, shapes, headers, footers, stamps, and layout.

python-docx's Document.save() silently drops unsupported elements
(textboxes, VML shapes, positioned frames), destroying complex layouts.
This module avoids that by only modifying <w:t> text nodes in the XML.

CRITICAL FIX (namespace preservation):
  etree.fromstring() + etree.tostring() causes lxml to REBUILD namespace
  declarations — moving them from the root element to individual elements,
  renaming prefixes (w: → ns0:, etc.), and reordering them. Word then
  fails to recognise many elements, causing text to jump positions and
  the entire layout to collapse (everything centers / reflows).

  Fix: use etree.parse(BytesIO(...)) + tree.write() instead.
  _ElementTree.write() serialises using the document's ORIGINAL namespace
  map, preserving every prefix and declaration exactly as Word wrote them.
"""
import json
import zipfile
import difflib
from io import BytesIO
from pathlib import Path
from typing import Union

from lxml import etree

WORD_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
XML_NS  = 'http://www.w3.org/XML/1998/namespace'

# XMLParser that never strips whitespace or resolves entities,
# matching Word's own serialisation behaviour as closely as possible.
_PARSER = etree.XMLParser(
    remove_blank_text=False,
    resolve_entities=False,
    recover=True,           # tolerate minor malformations Word files sometimes have
)


# ─────────────────────────────────────────────────────────────
# XML helpers: locate paragraphs, tables, cells
# ─────────────────────────────────────────────────────────────

def _get_body_paragraphs(root):
    """
    Get all direct <w:p> children of <w:body>.
    Matches python-docx's doc.paragraphs ordering exactly.
    """
    body = root.find(f'{{{WORD_NS}}}body')
    if body is None:
        return []
    return [child for child in body
            if etree.QName(child.tag).localname == 'p']


def _get_body_tables(root):
    """
    Get all direct <w:tbl> children of <w:body>.
    Matches python-docx's doc.tables ordering exactly.
    """
    body = root.find(f'{{{WORD_NS}}}body')
    if body is None:
        return []
    return [child for child in body
            if etree.QName(child.tag).localname == 'tbl']


def _get_table_cell(table_elem, row_idx, col_idx):
    """Get a specific <w:tc> element from a table."""
    rows = table_elem.findall(f'{{{WORD_NS}}}tr')
    if row_idx >= len(rows):
        return None
    cells = rows[row_idx].findall(f'{{{WORD_NS}}}tc')
    if col_idx >= len(cells):
        return None
    return cells[col_idx]


# ─────────────────────────────────────────────────────────────
# Text segment collection and replacement
# ─────────────────────────────────────────────────────────────

def _collect_text_segments(p_elem):
    """
    Collect <w:t> elements from a paragraph in reading order.

    Includes runs directly in <w:p> and inside <w:hyperlink>.
    Excludes textbox content (<w:txbxContent>) so we don't
    accidentally modify text in embedded shapes/textboxes.

    Returns: list of [t_element, text_string] pairs (mutable).
    """
    segments = []
    for child in p_elem:
        local = etree.QName(child.tag).localname
        if local == 'r':
            for t in child.findall(f'{{{WORD_NS}}}t'):
                segments.append([t, t.text or ''])
        elif local == 'hyperlink':
            for r in child.findall(f'{{{WORD_NS}}}r'):
                for t in r.findall(f'{{{WORD_NS}}}t'):
                    segments.append([t, t.text or ''])
    return segments


def _set_t_text(t_elem, text):
    """
    Set text on a <w:t> element.
    Also sets xml:space="preserve" if text has leading/trailing spaces
    (otherwise Word strips them on open).
    """
    t_elem.text = text
    if text and (text != text.strip()):
        t_elem.set(f'{{{XML_NS}}}space', 'preserve')


def _replace_substring_in_segments(segments, old_sub, new_sub):
    """
    Find old_sub across text segments and replace with new_sub.
    Only modifies <w:t> text content — all run formatting stays intact.

    Returns True on success, False if old_sub not found.
    """
    if not old_sub:
        return False

    full_text = "".join(text for _, text in segments)
    pos = full_text.find(old_sub)
    if pos == -1:
        return False

    end_pos = pos + len(old_sub) - 1

    # Build character → (segment_index, char_index) mapping
    char_map = []
    for seg_idx, (_, text) in enumerate(segments):
        for char_idx in range(len(text)):
            char_map.append((seg_idx, char_idx))

    if not char_map or pos >= len(char_map) or end_pos >= len(char_map):
        return False

    start_seg, start_char = char_map[pos]
    end_seg, end_char     = char_map[end_pos]

    if start_seg == end_seg:
        # Replacement within a single run
        t_elem, text = segments[start_seg]
        new_text = text[:start_char] + new_sub + text[end_char + 1:]
        _set_t_text(t_elem, new_text)
        segments[start_seg][1] = new_text
    else:
        # Spans multiple runs
        # First run: keep prefix + insert new text
        t_elem, text = segments[start_seg]
        new_first = text[:start_char] + new_sub
        _set_t_text(t_elem, new_first)
        segments[start_seg][1] = new_first

        # Middle runs: clear text (run XML nodes remain for formatting)
        for i in range(start_seg + 1, end_seg):
            _set_t_text(segments[i][0], '')
            segments[i][1] = ''

        # Last run: keep suffix
        t_elem, text = segments[end_seg]
        new_last = text[end_char + 1:]
        _set_t_text(t_elem, new_last)
        segments[end_seg][1] = new_last

    return True


def _modify_paragraph_xml(p_elem, new_text):
    """
    Replace paragraph text with new_text using diff-based approach.
    Only modifies the exact characters that changed.
    All run formatting, paragraph properties, and XML structure are preserved.
    """
    segments = _collect_text_segments(p_elem)
    if not segments:
        return False

    old_text = "".join(text for _, text in segments)
    if old_text == new_text:
        return True  # Nothing to change

    # Compute diff → list of (old_substring, new_substring)
    sm = difflib.SequenceMatcher(None, old_text, new_text, autojunk=False)
    replacements = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'replace':
            replacements.append((old_text[i1:i2], new_text[j1:j2]))
        elif op == 'delete':
            replacements.append((old_text[i1:i2], ''))

    if replacements:
        # Apply in reverse order so positions stay valid
        for old_sub, new_sub in reversed(replacements):
            _replace_substring_in_segments(segments, old_sub, new_sub)

        # Verify result
        result = "".join(seg[0].text or '' for seg in segments)
        if result == new_text:
            return True

    # Fallback: write everything to first segment, clear rest
    _set_t_text(segments[0][0], new_text)
    for seg in segments[1:]:
        _set_t_text(seg[0], '')

    return True


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def apply_modifications(
    docx_path: str,
    modifications: Union[dict, str],
    output_dir: str = None,
    output_filename: str = None,
) -> str:
    """
    Apply JSON modifications directly to .docx XML, preserving ALL layout.

    Works at the ZIP/XML level — never uses python-docx Document model,
    so textboxes, images, shapes, stamps, headers, footers, and all
    positioned/floating elements are perfectly preserved.

    Args:
        docx_path:       path to the original .docx file
        modifications:   dict with "modifications" key, or JSON string
        output_dir:      directory to save the revised file
        output_filename: custom filename (default: {original}_Revised.docx)

    Returns:
        Path to the revised .docx file
    """
    docx_path = Path(docx_path).resolve()
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    # Parse modifications
    if isinstance(modifications, str):
        modifications = json.loads(modifications)

    mods_list = modifications.get("modifications", [])
    if not mods_list:
        raise ValueError("No modifications provided.")

    print(f"[Doc Surgery] Applying {len(mods_list)} modifications to {docx_path.name}...")

    # Output path
    if output_dir is None:
        output_dir = docx_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"{docx_path.stem}_Revised.docx"

    output_path = output_dir / output_filename

    applied = 0
    errors  = []

    with zipfile.ZipFile(str(docx_path), 'r') as zin:
        # ── Parse document.xml preserving ALL namespace declarations ──────
        #
        # KEY FIX: use etree.parse(BytesIO) + tree.write() instead of
        # etree.fromstring() + etree.tostring().
        #
        # etree.fromstring/tostring causes lxml to REBUILD namespace
        # declarations: it moves them from the root to individual elements,
        # renames prefixes (w: → ns0:, etc.), and reorders declarations.
        # Word then fails to recognise those elements → layout collapses.
        #
        # etree.parse() + _ElementTree.write() keeps the original namespace
        # map intact, producing byte-for-byte identical namespace handling
        # for every element that was NOT modified.
        doc_xml = zin.read('word/document.xml')
        tree    = etree.parse(BytesIO(doc_xml), _PARSER)
        root    = tree.getroot()

        paragraphs = _get_body_paragraphs(root)
        tables     = _get_body_tables(root)

        for mod in mods_list:
            element_id = mod.get("id", "")
            new_text   = mod.get("new_text", "")

            try:
                if element_id.startswith("Para_"):
                    para_idx = int(element_id.split("_")[1])
                    if para_idx < len(paragraphs):
                        _modify_paragraph_xml(paragraphs[para_idx], new_text)
                        applied += 1
                        print(f"  ✓ Modified {element_id}")
                    else:
                        errors.append(
                            f"Paragraph index {para_idx} out of range "
                            f"(max: {len(paragraphs) - 1})"
                        )

                elif element_id.startswith("Table_") and "_Cell_" in element_id:
                    parts     = element_id.split("_")
                    table_idx = int(parts[1])
                    row_idx   = int(parts[3])
                    col_idx   = int(parts[4])

                    if table_idx < len(tables):
                        cell = _get_table_cell(tables[table_idx], row_idx, col_idx)
                        if cell is not None:
                            cell_paras = cell.findall(f'{{{WORD_NS}}}p')
                            if cell_paras:
                                _modify_paragraph_xml(cell_paras[0], new_text)
                            applied += 1
                            print(f"  ✓ Modified {element_id}")
                        else:
                            errors.append(f"Table cell ({row_idx}, {col_idx}) not found")
                    else:
                        errors.append(
                            f"Table index {table_idx} out of range "
                            f"(max: {len(tables) - 1})"
                        )
                else:
                    errors.append(f"Unknown element ID format: {element_id}")

            except Exception as e:
                errors.append(f"Error modifying {element_id}: {str(e)}")

        # ── Serialise modified XML — namespace-safe ────────────────────────
        #
        # tree.write() re-uses the original _ElementTree's namespace context,
        # so every xmlns: declaration stays on the root element exactly as
        # Word originally wrote it.  This is the companion fix to parse above.
        buf = BytesIO()
        tree.write(buf, xml_declaration=True, encoding='UTF-8', standalone=True)
        modified_xml = buf.getvalue()

        # ── Write new ZIP: copy everything, replace only document.xml ─────
        with zipfile.ZipFile(str(output_path), 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == 'word/document.xml':
                    zout.writestr(item, modified_xml)
                else:
                    zout.writestr(item, zin.read(item.filename))

    print(f"[Doc Surgery] ✓ Saved revised document: {output_path}")
    print(f"[Doc Surgery]   Applied: {applied}/{len(mods_list)}, Errors: {len(errors)}")

    if errors:
        for err in errors:
            print(f"  ⚠️  {err}")

    return str(output_path)