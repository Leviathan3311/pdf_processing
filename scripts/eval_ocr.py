"""Score MinerU outputs against hand-typed ground truth.

Usage (any Python, stdlib only):
    python scripts/eval_ocr.py Testcase -o Testcase/output Testcase/output/up3200

For every <stem>.gt.txt in the folder and every output root given, scores
<root>/<stem>/<tier>/middle.json (falls back to result.md).

Text comes from ALL non-image blocks of middle.json in reading order, including `header`
blocks (MinerU's own markdown drops them) and table cells. Two metrics:
  CER  - character error rate, order-sensitive (penalises wrong reading order too)
  WErr - order-free word error: 1 - F1 of the word multisets (pure recognition quality)
Normalisation: HTML tags/entities and markdown syntax removed, dot leaders removed,
whitespace collapsed, Unicode NFC. Case and diacritics are kept.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def normalise(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)                          # HTML tags (tables)
    text = html.unescape(text)
    text = re.sub(r"^\s*\|?\s*-{3,}.*$", " ", text, flags=re.M)  # markdown table separator rows
    text = re.sub(r"[|#`*\\]", " ", text)                         # markdown syntax
    text = re.sub(r"(?:\.{2,}|…+)", " ", text)                     # dot leaders on forms
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def _collect(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    out: list[str] = []
    for item in content or []:
        if isinstance(item, dict):
            out += _collect(item.get("content"))
    return out


def middle_json_text(path: Path) -> str:
    return middle_text(json.loads(path.read_text(encoding="utf-8")))


def middle_text(data: dict) -> str:
    """All non-image text of a middle.json (dict), in reading order."""
    parts: list[str] = []
    for page in data.get("pages", []):
        for block in sorted(page.get("blocks", []), key=lambda b: b.get("index", 0)):
            if block.get("type") == "image":      # seals, signatures, handwriting crops
                continue
            parts += _collect(block.get("content"))
    return "\n".join(parts)


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    return levenshtein(pred, truth) / max(1, len(truth))


def word_error(pred: str, truth: str) -> float:
    p, t = Counter(pred.split()), Counter(truth.split())
    hit = sum((p & t).values())
    if not hit:
        return 1.0
    precision, recall = hit / sum(p.values()), hit / sum(t.values())
    return 1 - 2 * precision * recall / (precision + recall)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="Folder containing <stem>.gt.txt files")
    ap.add_argument("-o", "--out", nargs="+", default=None, help="Output roots (default: <folder>/output)")
    args = ap.parse_args()

    folder = Path(args.folder)
    roots = [Path(r) for r in args.out] if args.out else [folder / "output"]
    print(f"{'sample':<9} {'run':<22} {'tier':<9} {'CER':>7} {'WErr':>7}  chars")
    for gt in sorted(folder.glob("*.gt.txt")):
        stem = gt.name[: -len(".gt.txt")]
        truth = normalise(gt.read_text(encoding="utf-8"))
        for root in roots:
            for tier_dir in sorted(p for p in (root / stem).glob("*") if p.is_dir()):
                mj, md = tier_dir / "middle.json", tier_dir / "result.md"
                if mj.exists():
                    raw = middle_json_text(mj)
                elif md.exists():
                    raw = md.read_text(encoding="utf-8")
                else:
                    continue
                pred = normalise(raw)
                print(f"{stem[:8]:<9} {str(root):<22} {tier_dir.name:<9} "
                      f"{cer(pred, truth):7.2%} {word_error(pred, truth):7.2%}  {len(pred)}/{len(truth)}")


if __name__ == "__main__":
    main()
