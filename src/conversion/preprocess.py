"""Page image clean-up before OCR (CLAUDE.md section 3, step 1b) — the measurement steps assume a flat,
level, white page, which phone photos and old scans are not.

  0. paper     a phone photo of a sheet on a desk / patterned cloth: find the sheet (the region around the
               centre with the paper's own colour) and warp it flat; scans and screenshots are left alone
  1. flatten   divide by the estimated paper background: uneven light, grey / tinted paper → white paper,
               ink keeps its own colour (needed for bold / colour measurement and transparent crops)
  2. deskew    rotate so text lines are horizontal (angle that maximises the row-profile contrast)
  3. upscale   long edge to `long_edge` px (Lanczos): the main lever for Vietnamese diacritics (section 5.5)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


# Crop only a sheet that clearly sits on something else. Measured (2026-11-27): photos with a desk / cloth
# around the sheet 0.64–0.83 of the frame; scans, screenshots and photos filling the frame 0.88–1.0.
PAPER_MAX_AREA = 0.85
PAPER_MIN_FILL = 0.9          # the sheet fills its 4-corner outline (not a ragged region)


def find_paper(rgb: np.ndarray) -> np.ndarray | None:
    """Corners (4×2, full-resolution px, clockwise from top-left) of the sheet, or None when the image is
    already just the page. The paper colour is taken around the centre once the writing is closed away, so
    a coloured light or tinted paper is still "paper"; the sheet is the connected region of that colour."""
    h, w = rgb.shape[:2]
    s = 600 / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    k = max(7, int(max(sh, sw) / 50) | 1)                       # wider than a pen stroke
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    lab = cv2.cvtColor(cv2.GaussianBlur(closed, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    cy, cx = sh // 2, sw // 2
    core = lab[cy - sh // 6:cy + sh // 6 + 1, cx - sw // 6:cx + sw // 6 + 1].reshape(-1, 3)
    paper = np.median(core, axis=0)
    dist = np.linalg.norm(lab - paper, axis=-1)
    spread = float(np.percentile(np.linalg.norm(core - paper, axis=-1), 90))   # the paper's own variation
    otsu, _ = cv2.threshold(np.clip(dist, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu splits paper from background; never tighter than the paper's own variation (uneven light).
    # Measured: a cap at 4 × that variation cut the far side of a sheet lit from one side
    mask = (dist < max(otsu, 2.5 * spread, 12.0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    _, labels = cv2.connectedComponents(mask)
    if labels[cy, cx] == 0:
        return None
    cnts, _ = cv2.findContours((labels == labels[cy, cx]).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    sheet = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(sheet)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        return None
    area = cv2.contourArea(quad)
    if area / (sh * sw) > PAPER_MAX_AREA or cv2.contourArea(sheet) / max(1.0, area) < PAPER_MIN_FILL:
        return None
    centre = quad.mean(axis=0)
    quad = centre + (quad - centre) * 0.99                       # keep the paper's shadowed edge out
    order = np.argsort(np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0]))   # clockwise from left-top
    quad = quad[order]
    start = int(np.argmin(quad.sum(axis=1)))                     # top-left: smallest x + y
    return np.roll(quad, -start, axis=0) / s


def warp_paper(rgb: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """The sheet seen straight on: a rectangle as wide / tall as the longer opposite edges."""
    tl, tr, br, bl = quad.astype(np.float32)
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    m = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dst)
    flat = cv2.warpPerspective(rgb, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # notes are written to the very edge of the sheet; a white margin like a scan's keeps the last line
    # inside the Word page instead of pushing it onto a second one
    pad = int(0.03 * max(width, height))
    paper = tuple(int(v) for v in np.median(flat.reshape(-1, 3), axis=0))
    return cv2.copyMakeBorder(flat, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=paper)


def flatten_background(rgb: np.ndarray) -> np.ndarray:
    """Paper → white everywhere, whatever the lighting: background = closing (removes text) + blur."""
    h, w = rgb.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    k = max(15, int(max(small.shape[:2]) / 40) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = rgb.astype(np.float32) / np.maximum(bg, 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def skew_angle(rgb: np.ndarray, max_deg: float = 5.0, step: float = 0.2) -> float:
    """Rotation (degrees) that makes text rows sharpest: max variance of the horizontal ink profile."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = 1200 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = ink.shape
    centre = (w / 2, h / 2)
    best, best_score = 0.0, -1.0
    for angle in np.arange(-max_deg, max_deg + 1e-6, step):
        rot = cv2.warpAffine(ink, cv2.getRotationMatrix2D(centre, angle, 1.0), (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best, best_score = float(angle), score
    return best


def rotate(rgb: np.ndarray, angle: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def prepare(src: Path, dst: Path, long_edge: int = 3200, clean: bool = False) -> float:
    """Clean `src` into `dst` (JPEG). Returns the skew angle that was corrected (degrees).
    `clean`: a page rendered from a PDF text layer — already flat and straight, only resized."""
    rgb = np.array(ImageOps.exif_transpose(Image.open(src)).convert("RGB"))
    angle = 0.0
    if not clean:
        quad = find_paper(rgb)
        if quad is not None:
            rgb = warp_paper(rgb, quad)
        rgb = flatten_background(rgb)
        angle = skew_angle(rgb)
        if abs(angle) >= 0.2:
            rgb = rotate(rgb, angle)
    img = Image.fromarray(rgb)
    scale = long_edge / max(img.size)
    if scale > 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    img.save(dst, quality=95)
    return angle


def render_pdf_page(page, dst: Path, long_edge: int = 3200, scan: bool = True) -> Path:
    """A PDF page as an image. A scanned page is rendered at the resolution of its own picture (the scan's
    pixels, nothing invented; `prepare` then upscales with Lanczos exactly as for an image file); a text
    page is vector, so it is rendered straight at the OCR size."""
    rect = page.rect
    dpi = long_edge / max(rect.width, rect.height) * 72
    if scan:
        infos = [i for i in page.get_image_info() if i.get("width")]
        if infos:
            big = max(infos, key=lambda i: (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]))
            bw, bh = big["bbox"][2] - big["bbox"][0], big["bbox"][3] - big["bbox"][1]
            if bw > 0 and bh > 0:
                native = max(big["width"] / bw, big["height"] / bh) * 72
                dpi = min(dpi, max(150.0, native))
    page.get_pixmap(dpi=round(dpi), alpha=False).save(str(dst))
    return dst
