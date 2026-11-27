"""Finding the sheet of paper in a phone photo (preprocess step 0)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conversion.preprocess import find_paper, warp_paper  # noqa: E402


def _page(h=900, w=640, tint=(255, 255, 255)):
    page = np.full((h, w, 3), tint, np.uint8)
    for y in range(60, h - 60, 45):
        cv2.putText(page, "dong chu viet tay", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 40), 2)
    return page


def _photo_of(page, size=(1600, 1200)):
    """The page warped onto a busy colourful cloth, a little tilted, like a phone photo."""
    rng = np.random.default_rng(0)
    h, w = size
    bg = np.zeros((h, w, 3), np.uint8)
    for _ in range(60):                                            # saturated print, like a cartoon cloth
        hsv = np.uint8([[[rng.integers(0, 180), rng.integers(150, 256), rng.integers(60, 256)]]])
        colour = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        cv2.circle(bg, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(30, 160)),
                   tuple(int(v) for v in colour), -1)
    ph, pw = page.shape[:2]
    corners = np.float32([[260, 330], [880, 300], [920, 1250], [230, 1290]])
    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]), corners)
    warped = cv2.warpPerspective(page, m, (w, h))
    inside = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), m, (w, h)) > 0
    bg[inside] = warped[inside]
    return bg, corners


def test_find_paper_on_cloth_under_coloured_light():
    photo, corners = _photo_of(_page(tint=(225, 190, 245)))       # purple-lit sheet
    quad = find_paper(photo)
    assert quad is not None
    assert np.abs(quad - corners).max() < 25                        # px on a 1600 px photo
    flat = warp_paper(photo, quad)
    assert 1.3 < flat.shape[0] / flat.shape[1] < 1.5                 # the page's own aspect is back
    assert flat[2, 2].tolist() == flat[-3, -3].tolist()              # a plain paper margin all round


def test_find_paper_leaves_a_full_frame_page_alone():
    assert find_paper(_page()) is None
