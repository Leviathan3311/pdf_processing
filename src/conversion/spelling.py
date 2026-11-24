"""Constrained Vietnamese diacritic correction for OCR output (CLAUDE.md section 5.5, step 2).

A seq2seq Vietnamese corrector (default: protonx-models/protonx-legal-tc, non-commercial licence) proposes
a rewrite of each sentence. A proposed change is applied ONLY when:
  - the word keeps the same letters once accents are removed (đ counts as d), and keeps its upper/lower case;
  - the word is not locked (contains a digit or one of / & ( ) — codes, dates, numbers).
Everything else the model writes is discarded, so it cannot invent, drop or reorder words.

Works directly on MinerU `middle.json` (dict): text spans and table cells are corrected in place;
whitespace and line breaks inside a span are preserved; image blocks are never touched.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

LOCK_CHARS = set("/&()")
SKIP_TYPES = {"image", "image_body", "equation", "equation_inline", "interline_equation", "inline_equation"}
_CELL_RE = re.compile(r"(<t[dh][^>]*>)(.*?)(</t[dh]>)", re.S)


def skeleton(word: str, keep_case: bool = True) -> str:
    """Word without diacritics; đ/Đ → d/D."""
    word = word.replace("đ", "d").replace("Đ", "D")
    base = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return base if keep_case else base.lower()


def is_locked(word: str) -> bool:
    """Codes, numbers, dates are never corrected."""
    return any(c.isdigit() for c in word) or any(c in LOCK_CHARS for c in word)


def accept(original: str, proposed: str) -> bool:
    return (original != proposed and not is_locked(original)
            and skeleton(original) == skeleton(proposed))


def merge(original: str, rewrite: str) -> tuple[str, list[tuple[str, str]]]:
    """Apply accepted word changes from `rewrite` to `original`, keeping original whitespace exactly."""
    parts = re.split(r"(\s+)", original)                 # words at even indices, whitespace at odd
    words = parts[0::2]
    proposed = rewrite.split()
    changes: list[tuple[str, str]] = []
    matcher = difflib.SequenceMatcher(None, words, proposed, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                old, new = words[i1 + k], proposed[j1 + k]
                if accept(old, new):
                    words[i1 + k] = new
                    changes.append((old, new))
    parts[0::2] = words
    return "".join(parts), changes


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Seq2seq correctors are trained on single sentences; long inputs make them hallucinate."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, cur = [], ""
    for piece in re.split(r"(?<=[.;:])\s+", text):
        if cur and len(cur) + len(piece) + 1 > max_chars:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    return chunks + ([cur] if cur else [])


class Corrector:
    """Loads the seq2seq model once; `rewrite` corrects a batch of strings."""

    def __init__(self, model_dir: str | Path, beams: int = 4, batch: int = 16, device: str = "cuda"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), dtype=dtype).to(device).eval()
        self.device, self.beams, self.batch = device, beams, batch

    def rewrite(self, texts: list[str]) -> list[str]:
        pieces, owners = [], []
        for i, text in enumerate(texts):
            for sentence in split_sentences(text):
                pieces.append(sentence)
                owners.append(i)
        outputs: list[str] = []
        for k in range(0, len(pieces), self.batch):
            chunk = pieces[k:k + self.batch]
            caps = [s.isupper() for s in chunk]           # all-caps lines: correct in lower case
            enc = self.tok([s.lower() if c else s for s, c in zip(chunk, caps)], return_tensors="pt",
                           padding=True, truncation=True, max_length=256).to(self.device)
            with self.torch.no_grad():
                out = self.model.generate(**enc, num_beams=self.beams, max_new_tokens=256)
            for text, c in zip(self.tok.batch_decode(out, skip_special_tokens=True), caps):
                outputs.append(text.upper() if c else text)
        result = [""] * len(texts)
        for i, text in zip(owners, outputs):
            result[i] = f"{result[i]} {text}".strip()
        return result


@dataclass
class _Slot:
    """One correctable string inside middle.json, with a setter to write it back."""
    text: str
    write: callable
    where: str


@dataclass
class Report:
    changes: list[dict] = field(default_factory=list)


def _slots(middle: dict) -> list[_Slot]:
    slots: list[_Slot] = []

    def visit(node, where: str) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item, where)
            return
        if not isinstance(node, dict):
            return
        kind, content = node.get("type"), node.get("content")
        if kind in SKIP_TYPES:                            # images and LaTeX are never touched
            return
        if kind == "table_body" and isinstance(content, str):
            def write_html(new_cells, node=node):
                it = iter(new_cells)
                node["content"] = _CELL_RE.sub(lambda m: m.group(1) + next(it) + m.group(3), node["content"])
            cells = [m.group(2) for m in _CELL_RE.finditer(content)]
            fixed: list[str] = list(cells)
            for idx, cell in enumerate(cells):
                def write_cell(value, idx=idx, fixed=fixed, write_html=write_html):
                    fixed[idx] = value
                    write_html(fixed)
                slots.append(_Slot(cell, write_cell, f"{where}/cell{idx}"))
            return
        if isinstance(content, str):
            def write_text(value, node=node):
                node["content"] = value
            slots.append(_Slot(content, write_text, where))
        else:
            visit(content, where)

    for page in middle.get("pages", []):
        for block in page.get("blocks", []):
            visit(block, f"p{page.get('page_idx', 0)}/b{block.get('index')}:{block.get('type')}")
    return slots


def correct_middle_json(middle: dict, corrector: Corrector) -> Report:
    """Correct all text spans and table cells of `middle` in place; return the list of applied changes."""
    return correct_many([middle], corrector)[0]


def correct_many(middles: list[dict], corrector: Corrector) -> list[Report]:
    """Correct several documents in one batched model pass (GPU batches span documents)."""
    owned = [(k, s) for k, middle in enumerate(middles) for s in _slots(middle)
             if s.text.strip() and "<" not in s.text]
    rewrites = corrector.rewrite([s.text for _, s in owned])
    reports = [Report() for _ in middles]
    for (k, slot), rewrite in zip(owned, rewrites):
        new_text, changes = merge(slot.text, rewrite)
        if changes:
            slot.write(new_text)
            reports[k].changes += [{"where": slot.where, "from": a, "to": b} for a, b in changes]
    return reports
