"""Lean MinerU runner used by the pipeline: one middle.json per input, nothing else.

Run inside the MinerU venv:
    .venv-mineru\\Scripts\\python scripts\\mineru_parse.py <files or dirs> -o <out_dir> [--tier standard]

Writes <out_dir>/<stem>.json (MinerU middle.json, image_base64 kept so crops of seals / signatures survive).
Models load once for the whole batch; no warm-up pass, no markdown / docx renders.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mineru.parser import parse

SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tier", default="standard")
    args = ap.parse_args()

    files: list[Path] = []
    for p in map(Path, args.inputs):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in SUFFIXES) if p.is_dir() else [p]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for src in files:
        t0 = time.perf_counter()
        try:
            middle = parse(src, tier=args.tier).middle_json
            (out / f"{src.stem}.json").write_text(json.dumps(middle.to_dict(skip_defaults=True), ensure_ascii=False),
                                                  encoding="utf-8")
            print(f"ok {src.name} {time.perf_counter() - t0:.2f}s", flush=True)
        except Exception as exc:  # report and continue with the rest of the batch
            failed += 1
            print(f"FAILED {src.name}: {type(exc).__name__}: {exc}", flush=True)
    sys.exit(1 if failed == len(files) else 0)


if __name__ == "__main__":
    main()
