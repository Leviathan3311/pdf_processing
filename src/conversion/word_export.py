"""Microsoft Word as the reference renderer: DOCX → PDF through PowerShell COM (nothing to install),
one Word instance per batch, plus a side-by-side comparison image for debugging."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

_PS_BATCH = r"""
$pairs = Get-Content -LiteralPath '{listing}' -Encoding UTF8
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {{
  foreach ($line in $pairs) {{
    if (-not $line) {{ continue }}
    $src, $dst = $line -split "`t"
    $doc = $word.Documents.Open($src, $false, $true)
    $doc.SaveAs([ref]$dst, [ref]17)
    $doc.Close([ref]0)
  }}
}} finally {{ $word.Quit() }}
"""


def docx_to_pdf_batch(pairs: list[tuple[Path, Path]]) -> None:
    """Export every (docx, pdf) pair with ONE Word instance (starting Word costs ~3 s each time)."""
    if not pairs:
        return
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "pairs.txt"
        listing.write_text("\n".join(f"{Path(s).resolve()}\t{Path(d).resolve()}" for s, d in pairs),
                           encoding="utf-8-sig")
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                        _PS_BATCH.format(listing=str(listing))], check=True, capture_output=True)


def docx_to_pdf(docx: Path, pdf: Path) -> None:
    docx_to_pdf_batch([(docx, pdf)])


def compare_image(scan: Path, pdf: Path, out: Path, height: int = 1400, page_no: int = 0) -> None:
    """Original page image (left) | that page rendered by Word (right), same height."""
    left = cv2.imread(str(scan))
    doc = fitz.open(str(pdf))
    if page_no >= len(doc):
        return
    page = doc[page_no]
    pix = page.get_pixmap(dpi=150)
    right = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)

    def fit(img):
        scale = height / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), height), interpolation=cv2.INTER_AREA)

    gap = np.full((height, 12, 3), 128, np.uint8)
    cv2.imwrite(str(out), np.hstack([fit(left), gap, fit(right)]))


def pdf_page_count(pdf: Path) -> int:
    return len(fitz.open(str(pdf)))
