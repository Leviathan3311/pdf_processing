"""Page image clean-up before OCR (CLAUDE.md section 3, step 1b) — the measurement steps assume a flat,
level, white page, which phone photos and old scans are not.

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
