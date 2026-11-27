"""Handwriting re-read: a local VLM (Qwen3.5-9B GGUF via llama-server) reads again the MinerU blocks that are
handwritten, MinerU keeps the layout (docs/handwriting-ocr/DESIGN.md).

    regions = find_regions(middle_page, page_image, mode)      # which blocks (DESIGN 3.2)
    with VlmServer(...) as server: texts = read_regions(regions, server)    # parallel slots (DESIGN 4)
    apply(region, text)                                          # checks + choice + write-back (DESIGN 5)

Signals used (DESIGN 3.2): T1 coloured-ink block on a printed page (not red: red there is a seal or stamp, which
stays a picture), T2 every prose block of a pen-written page (red pen titles included).
T3 (`sub_type: text_image`) is not used: MinerU 4.2.3 returned only `seal` on the 14 test pages.
T4 (black-ink handwriting) and T5 (table cells) are not implemented. `force` re-reads every prose block.
A region whose VLM text fails a check keeps today's result (MinerU text, or the cropped picture).
"""
from __future__ import annotations

import base64
import json
import re
import socket
import subprocess
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PROMPT = (
    "Chép lại chính xác toàn bộ chữ viết tay trong ảnh, giữ nguyên tiếng Việt có dấu, chữ viết tắt và các con số. "
    "Xuống dòng đúng như trong ảnh. Công thức hoặc ký hiệu toán viết bằng LaTeX đặt trong $...$. "
    "Chỉ trả về nội dung, không giải thích. "
    "Nếu ảnh là chữ ký thì trả về <SIGNATURE>; nếu không có chữ thì trả về <EMPTY>."
)
MAX_PIXELS = 1280 * 28 * 28            # same pixel budget as the benchmark (DESIGN 4)
PAD_PX = 6                             # around MinerU's box, on the 3200 px page image
MIN_HEIGHT_PX = 24                     # smaller boxes are ticks / dashes (DESIGN 3.4)
PROSE_SHARE = 0.6                     # an "equation" block that is mostly \text{...} prose is a prose block
REVIEW_DISTANCE = 0.3                  # MinerU and VLM both valid but this far apart → needs_review
SKIP_TYPES = {"image", "table"}
EXPLANATION = re.compile(r"^\s*(đây là|văn bản trong ảnh|nội dung (trong|của) ảnh|here is|the text)", re.I)


@dataclass
class Region:
    unit: str                          # work-folder key of the page
    block: dict                        # the middle.json block (written in place)
    signal: str                        # T1 | T2 | force
    image: bytes                       # PNG crop
    expected_chars: float              # from the measured lines: width / (0.5 × letter height)
    line_units: float                  # bbox width / line height × lines (length check)
    mineru_text: str
    colored: bool
    text: str | None = None            # VLM output
    finish: str | None = None
    seconds: float = 0.0
    verdict: str = ""                  # accepted | fallback:<reason> | signature
    review: bool = False


@dataclass
class HwStats:
    sent: int = 0
    accepted: int = 0
    fallback: int = 0
    review: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.sent}/{self.accepted}/{self.fallback}/{self.review}" if self.sent else "—"


# ---------------------------------------------------------------- 3. which blocks

def _shown(block: dict) -> tuple[str, str]:
    """(all text the block shows, the prose part of it): math through its display text, \\text{} as prose."""
    from .layout import _segments
    from .math_omml import display_text
    shown, prose = [], []
    for kind, value in _segments(block.get("content"), block.get("type", "text")):
        if kind == "math":
            shown.append(display_text(value))
            prose += re.findall(r"\\(?:text|mathrm|textrm)\s*\{([^{}]*)\}", value)
        else:
            shown.append(value)
            prose.append(value)
    return " ".join(shown), " ".join(prose)


def _alnum(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _prose_block(block: dict) -> bool:
    """Prose, not a formula: VLM reads prose; MinerU's LaTeX for matrices / fractions stays."""
    if block.get("type") in SKIP_TYPES or not block.get("bbox"):
        return False
    from .layout import _segments
    from .math_omml import height_em
    segments = _segments(block.get("content"), block.get("type", "text"))
    if any(kind == "math" and height_em(value) > 1.5 for kind, value in segments):
        return False                                     # a fraction / matrix: MinerU's LaTeX stays (DESIGN 3.4)
    shown, prose = _shown(block)
    return _alnum(shown) > 0 and _alnum(prose) >= PROSE_SHARE * _alnum(shown)


def _crop_png(rgb: np.ndarray, box: tuple[int, int, int, int]) -> bytes:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, box[0] - PAD_PX), max(0, box[1] - PAD_PX), min(w, box[2] + PAD_PX), min(h, box[3] + PAD_PX)
    crop = rgb[y0:y1, x0:x1]
    scale = min(1.0, (MAX_PIXELS / max(1, crop.shape[0] * crop.shape[1])) ** 0.5)
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


def _blocks(blocks: list[dict]) -> list[dict]:
    """Blocks as layout sees them (list items on their own), but the original dicts, so writes stick."""
    out = []
    for b in blocks:
        items = [c for c in (b.get("content") or []) if isinstance(c, dict) and c.get("bbox")]
        out += items if b.get("type") == "list" and items else [b]
    return out


def _red_ink(color: str | None) -> bool:
    """Red is the colour of seals and stamps on official letters ("HỎA TỐC", the round seal's text); pen fill-ins
    are blue. Measured on 5 printed pages: seals r−grey +18…+55 with b < 0, blue notes b−grey +17…+29."""
    if not color:
        return False
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return r > max(g, b)


def find_regions(middle_page: dict, image_path: Path, unit: str, mode: str = "auto") -> list[Region]:
    """Blocks of one page that the VLM should read again (DESIGN 3.2). `mode`: auto | force."""
    from .measure import ink_reference, is_pen_page, measure_block
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return []
    rgb, gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blocks = _blocks(middle_page.get("blocks", []))

    def px(b):
        return int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)

    text_boxes = [px(b["bbox"]) for b in blocks if b.get("type") not in SKIP_TYPES and b.get("bbox")]
    ink_ref = ink_reference(rgb, gray, text_boxes)
    pen = is_pen_page(ink_ref)
    regions = []
    for b in blocks:
        if not _prose_block(b):
            continue
        box = px(b["bbox"])
        if box[3] - box[1] < MIN_HEIGHT_PX:
            continue
        m = measure_block(rgb, gray, box, ink_ref=ink_ref, pen_page=pen)
        signal = ("force" if mode == "force" else "T2" if pen
                  else "T1" if m.colored_ink and not _red_ink(m.color) else None)   # red on print: a seal stays a picture
        if signal is None or not m.lines:
            continue
        shown, _ = _shown(b)
        if _alnum(shown) <= 2 and not m.colored_ink:     # a tick or a dash read as a character
            continue
        units = sum((ln.x1 - ln.x0) / max(1.0, ln.letter_h or (ln.y1 - ln.y0)) for ln in m.lines)
        line_h = float(np.median([ln.y1 - ln.y0 for ln in m.lines]))
        regions.append(Region(unit, b, signal, _crop_png(rgb, box), expected_chars=2.0 * units,
                              line_units=len(m.lines) * (box[2] - box[0]) / max(1.0, line_h),
                              mineru_text=shown, colored=m.colored_ink))
    return regions


# ---------------------------------------------------------------- 4. calling the VLM

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VlmServer:
    """`llama-server` with the vision projector, started in the background (DESIGN 4)."""

    def __init__(self, exe: Path, model: Path, mmproj: Path, slots: int = 4, ctx_per_slot: int = 4096,
                 log_path: Path | None = None):
        self.exe, self.model, self.mmproj, self.slots = Path(exe), Path(model), Path(mmproj), slots
        self.ctx = ctx_per_slot * slots
        self.port = _free_port()
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    @property
    def available(self) -> bool:
        return self.exe.exists() and self.model.exists() and self.mmproj.exists()

    def start(self) -> "VlmServer":
        args = [str(self.exe), "-m", str(self.model), "--mmproj", str(self.mmproj), "--host", "127.0.0.1",
                "--port", str(self.port), "--parallel", str(self.slots), "-c", str(self.ctx), "-fa", "on",
                "--jinja", "--reasoning-budget", "0", "-ngl", "999", "--no-webui"]
        log = open(self.log_path, "w", encoding="utf-8") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return self

    def wait_ready(self, timeout: float = 180.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.proc is None or self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def read(self, png: bytes, max_tokens: int) -> tuple[str, str]:
        body = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                {"type": "text", "text": PROMPT}]}],
            "temperature": 0, "repeat_penalty": 1.05, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read())
        choice = out["choices"][0]
        return (choice["message"].get("content") or "").strip(), choice.get("finish_reason") or ""


def read_regions(regions: list[Region], server: VlmServer) -> None:
    """All regions of the batch through one queue, `slots` requests at a time (DESIGN 6)."""
    def one(r: Region) -> None:
        t = time.perf_counter()
        limit = max(64, int(1.5 * r.expected_chars))
        try:
            r.text, r.finish = server.read(r.image, limit)
            if r.finish == "length":                      # the estimate from the measured lines was short:
                r.text, r.finish = server.read(r.image, 2 * limit)   # one more try, still bounded (repeats)
        except Exception as exc:                          # a failed request falls back like a failed check
            r.text, r.finish = None, f"error: {type(exc).__name__}"
        r.seconds = time.perf_counter() - t
    with ThreadPoolExecutor(max_workers=server.slots) as pool:
        list(pool.map(one, regions))


# ---------------------------------------------------------------- 5. checks, choice, write-back

_ALLOWED = [(0x0000, 0x024F), (0x0300, 0x036F), (0x0370, 0x03FF), (0x1E00, 0x1EFF), (0x2000, 0x2BFF),
            (0x2460, 0x24FF), (0x3000, 0x303F), (0xFF00, 0xFF0F)]


def _foreign_share(text: str) -> float:
    """Share of characters outside Latin / Vietnamese / Greek / digits / punctuation / math (IPA, CJK …)."""
    chars = [ch for ch in text if not ch.isspace()]
    bad = sum(1 for ch in chars if not any(a <= ord(ch) <= b for a, b in _ALLOWED) or 0x0250 <= ord(ch) <= 0x02AF)
    return bad / max(1, len(chars))


def _repeats(text: str, n: int = 3) -> int:
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return max((grams.count(g) for g in set(grams)), default=0)


def _plain(text: str) -> str:
    """Shown characters of a VLM answer: $…$ math through its display text."""
    from .math_omml import display_text
    return re.sub(r"\$\$?(.+?)\$\$?", lambda m: display_text(m.group(1)), text, flags=re.S)


_MATH = re.compile(r"\$\$?(.+?)\$\$?", re.S)
_SPELLED = re.compile(r"(?<![\\A-Za-z])(?:[A-Za-z] ){2,}[A-Za-z](?![A-Za-z])")


def check(text: str | None, finish: str | None, line_units: float, reference: str = "") -> str:
    """'' when `text` passes DESIGN 5.1, else the reason.

    Length: against MinerU's own reading of the same block when it has one (relative to the page, like the
    layout rules), else against the box (chars per line height). Measured on 26 regions of a pen page: correct
    readings sit at 3.0–4.4 chars per line height, above the design's first guess of 3."""
    if not text or text.strip() in ("<EMPTY>",):
        return "empty"
    if finish == "length":
        return "max_tokens"
    if finish and finish.startswith("error"):
        return finish
    if EXPLANATION.match(text):
        return "explanation"
    plain = _plain(text)
    n = len(plain.replace("\n", "").replace(" ", ""))
    ref = len(reference.replace("\n", "").replace(" ", ""))
    ratio = n / ref if ref >= 4 else n / max(1.0, line_units)
    if not 0.33 <= ratio <= (3.0 if ref >= 4 else 6.0):
        return f"length {ratio:.2f}"
    if re.search(r"\\[A-Za-z]+", _MATH.sub("", text)):
        return "latex outside $"                          # would be printed as raw LaTeX
    if any(_SPELLED.search(m) for m in _MATH.findall(text)):
        return "word as math"                             # "s c a l e": prose set as single-letter variables
    if _repeats(plain) > 3:
        return "repeats"
    if _foreign_share(plain) >= 0.05:
        return "script"
    return ""


def _distance(a: str, b: str) -> float:
    """Character error rate of `b` against `a` (normalised whitespace)."""
    a, b = " ".join(a.split()), " ".join(b.split())
    if not a:
        return 1.0 if b else 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


_BARE_SCRIPT = re.compile(r"\b[A-Za-z0-9]+(?:[_^]\{[^{}$]*\})+")   # "X_{proj}" written without $
_ROWS_ENV = re.compile(r"\\begin\{(gathered|aligned|array|cases|split|gather\*?|align\*?)\}(?:\{[^{}]*\})?"
                       r"(.*?)\\end\{\1\}", re.S)
_TEXT = re.compile(r"\\(?:text|mathrm|textrm)\s*\{([^{}]*)\}")
_DELIM = re.compile(r"\\(?:left|right)\s*(?:\\[{}]|[\[\]()|.])")


def _unwrap_rows(latex: str) -> str | None:
    """Handwritten lines that the VLM set as one multi-row environment (`\\begin{gathered} \\text{…} \\\\ …`)
    → one text line per row, leftover symbols as inline $…$. Layout would treat the environment as one tall
    formula and squeeze the whole block onto a single line. Real formulas (little \\text) stay as they are."""
    from .math_omml import display_text
    env = _ROWS_ENV.search(latex)
    if env is None:
        return None
    if _alnum(" ".join(_TEXT.findall(latex))) < PROSE_SHARE * _alnum(display_text(latex)):
        return None

    def as_text(piece: str) -> str:
        out = []
        for i, part in enumerate(_TEXT.split(_DELIM.sub("", piece).replace("&", " "))):
            part = part.strip()
            if part:
                out.append(part if i % 2 else f"${part}$")
        return " ".join(out)

    lines = [as_text(row) for row in re.split(r"\\\\", env.group(2)) if row.strip()]
    if not lines:
        return None
    before, after = as_text(latex[:env.start()]), as_text(latex[env.end():])
    lines[0] = f"{before} {lines[0]}".strip()
    lines[-1] = f"{lines[-1]} {after}".strip()
    return "\n".join(lines)


def _spans(text: str) -> list[dict]:
    """VLM answer → middle.json spans: prose as `text`, $…$ (and bare `X_{…}`) as `equation_inline`, marked `hw`."""
    out = []

    def add(kind: str, value: str) -> None:
        if value:
            out.append({"type": kind, "content": value, "hw": True, "src": "vlm"})

    for i, part in enumerate(re.split(r"\$\$?(.+?)\$\$?", text, flags=re.S)):
        if i % 2:
            rows = _unwrap_rows(part)
            if rows is not None:
                out.extend(_spans(rows))
            else:
                add("equation_inline", part.strip())
            continue
        pos = 0
        for m in _BARE_SCRIPT.finditer(part):
            add("text", part[pos:m.start()])
            add("equation_inline", m.group(0))
            pos = m.end()
        add("text", part[pos:])
    return out


def apply(r: Region) -> None:
    """Choose MinerU or VLM for the region (DESIGN 5.2) and write the choice into the block (DESIGN 5.3)."""
    if r.text and r.text.strip() == "<SIGNATURE>":
        r.verdict = "signature"                          # keep today's result (a coloured block stays a picture)
        return
    text = unicodedata.normalize("NFC", r.text or "")
    reason = check(text, r.finish, r.line_units, r.mineru_text)
    if reason:
        r.verdict = f"fallback:{reason}"
        return
    if r.mineru_text.strip() and not check(r.mineru_text, "stop", r.line_units):
        r.review = _distance(_plain(text), r.mineru_text) > REVIEW_DISTANCE
    if r.review and r.signal == "T1":
        # a short pen note on a printed page ("Đồng ý" + flourish): two readers disagree → the picture stays
        r.verdict, r.review = "fallback:disagree", False
        return
    block = r.block
    if block.get("type") in ("equation", "interline_equation"):
        block["type"] = "text"                          # MinerU had set this prose line as a formula
    block["content"] = _spans(text)
    block["hw"] = True
    if r.review:
        block["needs_review"] = True
    r.verdict = "accepted"


def stats_of(regions: list[Region]) -> HwStats:
    s = HwStats(sent=len(regions), seconds=sum(r.seconds for r in regions))
    s.accepted = sum(r.verdict == "accepted" for r in regions)
    s.fallback = s.sent - s.accepted
    s.review = sum(r.review for r in regions)
    s.notes = [f"{r.unit} b{r.block.get('index')}: {r.verdict}" for r in regions if r.verdict != "accepted"]
    return s
