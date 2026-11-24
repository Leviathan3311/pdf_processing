"""Benchmark a local LLM for the diacritic-fix stage against hand-typed ground truth.

Usage (main project venv, needs torch+CUDA and transformers):
    python scripts/bench_llm.py --model models/protonx-legal-tc --middle <run>/<stem>/standard/middle.json \
        --gt Testcase/<stem>.gt.txt

Method (the only one that worked with Qwen3-4B; asking for a "wrong -> right" list was ignored):
  1. Split middle.json into text units: each non-image block, each table cell.
  2. LLM rewrites each unit verbatim, fixing diacritics only (non-thinking, greedy).
  3. Code diffs the rewrite word by word and accepts a change ONLY if the unaccented skeleton is
     identical and the word is not locked (digits, / & ( )). Everything else is discarded.
  4. Word errors vs ground truth before/after, plus load time, VRAM, tokens/s.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ocr import normalise  # noqa: E402

SYSTEM_PLAIN = (
    "Bạn là công cụ sửa lỗi dấu tiếng Việt cho văn bản OCR. Viết lại NGUYÊN VĂN đoạn văn, chỉ sửa dấu thanh, "
    "dấu mũ, dấu móc, d/đ khi sai. Không thêm, bớt hay đổi từ. Không giải thích."
)
SYSTEM_FEWSHOT = (
    "Văn bản dưới đây là kết quả OCR từ văn bản hành chính tiếng Việt, THƯỜNG CÓ LỖI DẤU: sai dấu thanh "
    "(sắc/huyền/hỏi/ngã/nặng), thiếu hoặc sai dấu mũ/móc (a/ă/â, o/ô/ơ, u/ư, e/ê), d thay cho đ. "
    "Hãy đọc theo ngữ cảnh, tìm các từ sai dấu và sửa lại. Giữ nguyên số lượng từ, thứ tự từ, chữ cái gốc, "
    "dấu câu, số và mã hiệu. Chỉ trả về văn bản đã sửa, mỗi dòng đầu vào tương ứng một dòng đầu ra."
)
# Examples are invented, deliberately NOT taken from any Testcase sample (no answer leakage).
FEWSHOT = [
    ("Ủy ban nhân dân tỉnh đã tô chức hôi nghị tông kết công tác năm 2025.\nSở Giáo duc và Đào tao",
     "Ủy ban nhân dân tỉnh đã tổ chức hội nghị tổng kết công tác năm 2025.\nSở Giáo dục và Đào tạo"),
    ("Yêu câu các dơn vi báo cáo két quả thực hiên truớc ngày 15/12.\nGIÁM DÔC SỞ",
     "Yêu cầu các đơn vị báo cáo kết quả thực hiện trước ngày 15/12.\nGIÁM ĐỐC SỞ"),
]


def build_messages(variant: str, text: str) -> list[dict]:
    if variant == "plain":
        return [{"role": "system", "content": SYSTEM_PLAIN}, {"role": "user", "content": text}]
    msgs = [{"role": "system", "content": SYSTEM_FEWSHOT}]
    for user, assistant in FEWSHOT:
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return msgs + [{"role": "user", "content": text}]


def skeleton(word: str, keep_case: bool = False) -> str:
    word = word.replace("đ", "d").replace("Đ", "D")
    base = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return base if keep_case else base.lower()


def is_locked(word: str) -> bool:
    """Codes, numbers, dates: never corrected."""
    return any(c.isdigit() for c in word) or any(c in "/&()" for c in word)


def _collect(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    out: list[str] = []
    for item in content or []:
        if isinstance(item, dict):
            out += _collect(item.get("content"))
    return out


def text_units(middle_json: Path) -> list[str]:
    units: list[str] = []
    for page in json.loads(middle_json.read_text(encoding="utf-8"))["pages"]:
        for block in sorted(page["blocks"], key=lambda b: b.get("index", 0)):
            if block["type"] == "image":
                continue
            if block["type"] == "table":
                for part in block.get("content") or []:      # table_body (HTML) + caption / footnote text
                    raw = " ".join(_collect(part.get("content")))
                    if part.get("type") == "table_body":
                        units += [normalise(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", raw, re.S)]
                    else:
                        units.append(normalise(raw))
            else:
                units += [normalise(s) for s in _collect(block.get("content"))]
    return [u for u in units if u]


def constrained_merge(original: str, rewrite: str) -> tuple[str, list[str]]:
    a, b = original.split(), rewrite.split()
    result, changes = list(a), []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                x, y = a[i1 + k], b[j1 + k]
                if x != y and not is_locked(x) and skeleton(x, keep_case=True) == skeleton(y, keep_case=True):
                    result[i1 + k] = y
                    changes.append(f"{x}→{y}")
    return " ".join(result), changes


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Seq2seq correctors are trained on single sentences; long inputs make them hallucinate."""
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[.;:])\s+", text)
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    return chunks + ([cur] if cur else [])


def run_seq2seq(model_dir: str, units: list[str], beams: int, batch: int) -> tuple[list[str], int]:
    """Correct each sentence with an encoder-decoder corrector. All-caps lines are corrected in lower case
    and re-uppercased (these models are trained on sentence-case text)."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=torch.float16).to("cuda").eval()
    pieces, owners = [], []
    for i, unit in enumerate(units):
        for s in split_sentences(unit):
            pieces.append(s)
            owners.append(i)
    outputs, n_out = [], 0
    for k in range(0, len(pieces), batch):
        chunk = pieces[k:k + batch]
        caps = [s.isupper() for s in chunk]
        inputs = [s.lower() if c else s for s, c in zip(chunk, caps)]
        enc = tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, num_beams=beams, max_new_tokens=256)
        n_out += int((out != tok.pad_token_id).sum())
        for text, c in zip(tok.batch_decode(out, skip_special_tokens=True), caps):
            outputs.append(text.upper() if c else text)
    rewritten = [""] * len(units)
    for i, text in zip(owners, outputs):
        rewritten[i] = f"{rewritten[i]} {text}".strip()
    return rewritten, n_out


def word_errors(words: list[str], truth: list[str]) -> int:
    ops = difflib.SequenceMatcher(None, words, truth, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/protonx-legal-tc")
    ap.add_argument("--middle", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--prompt", default="fewshot", choices=["plain", "fewshot"],
                    help="plain: bare instruction; fewshot: error-focused instruction + 2 invented examples")
    ap.add_argument("--doc", action="store_true",
                    help="Send the whole document in one request (one line per unit) instead of unit by unit")
    ap.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode")
    ap.add_argument("--beams", type=int, default=4, help="Beam search width for seq2seq correctors")
    ap.add_argument("--batch", type=int, default=16, help="Sentences per batch for seq2seq correctors")
    args = ap.parse_args()

    truth = normalise(Path(args.gt).read_text(encoding="utf-8")).split()
    units = text_units(Path(args.middle))

    if AutoConfig.from_pretrained(args.model).is_encoder_decoder:
        t0 = time.perf_counter()
        rewritten, n_out = run_seq2seq(args.model, units, args.beams, args.batch)
        secs = time.perf_counter() - t0
        fixed, changes = [], []
        for unit, new in zip(units, rewritten):
            merged, ch = constrained_merge(unit, new)
            fixed.append(merged)
            changes += ch
        before, after = " ".join(units).split(), " ".join(fixed).split()
        print(f"seq2seq: {secs:.1f} s incl. load, {len(units)} units, beams={args.beams}, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        print(f"word errors vs GT: before {word_errors(before, truth)} -> after {word_errors(after, truth)} "
              f"(of {len(truth)} words)")
        print("accepted:", ", ".join(changes) or "(none)")
        return

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map="cuda", attn_implementation="sdpa"
    ).eval()
    print(f"load: {time.perf_counter() - t0:.1f} s, VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    budget = 4096 if args.think else 0

    def ask(text: str) -> tuple[str, int]:
        prompt = tok.apply_chat_template(build_messages(args.prompt, text), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=args.think)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=int(len(text) * 0.9) + 40 + budget, do_sample=False)
        new_ids = out[0, ids.input_ids.shape[1]:]
        answer = tok.decode(new_ids, skip_special_tokens=True)
        answer = answer.split("</think>")[-1].strip()          # drop reasoning when thinking is on
        return answer, len(new_ids)

    t1, n_out, fixed, changes = time.perf_counter(), 0, [], []
    if args.doc:
        answer, n_out = ask("\n".join(units))
        lines = answer.splitlines()
        if len(lines) != len(units):                            # line mapping lost: fall back to whole-text merge
            merged, changes = constrained_merge(" ".join(units), " ".join(lines))
            fixed = [merged]
        else:
            for unit, line in zip(units, lines):
                merged, ch = constrained_merge(unit, line)
                fixed.append(merged)
                changes += ch
    else:
        for unit in units:
            if len(unit.split()) < 2:                           # single words / short cells: too little context
                fixed.append(unit)
                continue
            answer, n = ask(unit)
            n_out += n
            merged, ch = constrained_merge(unit, answer)
            fixed.append(merged)
            changes += ch
    secs = time.perf_counter() - t1

    before, after = " ".join(units).split(), " ".join(fixed).split()
    print(f"fix: {secs:.1f} s for {len(units)} units, {n_out} tokens out ({n_out / secs:.1f} tok/s), "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(f"word errors vs GT: before {word_errors(before, truth)} -> after {word_errors(after, truth)} "
          f"(of {len(truth)} words)")
    print("accepted:", ", ".join(changes) or "(none)")


if __name__ == "__main__":
    main()
