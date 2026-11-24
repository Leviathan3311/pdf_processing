"""Spike: run MinerU tiers on sample files and save every output for comparison.

Run inside the MinerU venv (not the main project venv):
    .venv-mineru\\Scripts\\python scripts\\spike_mineru.py Testcase --tiers basic standard

Output per input file and tier: <out>/<stem>/<tier>/{result.md, middle.json, content_list.json, mineru.docx}
plus <out>/timings.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mineru.parser import parse
from mineru.render.api import render
from mineru.render.contracts import RenderFormat

SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def collect_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in SUFFIXES)
        elif p.suffix.lower() in SUFFIXES:
            files.append(p)
    return files


def run_one(src: Path, tier: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = parse(src, tier=tier)
    parse_s = time.perf_counter() - t0

    mj = result.middle_json
    (out_dir / "result.md").write_text(result.markdown(), encoding="utf-8")
    # Keep image_base64 (ParseResult.to_json drops it for PDFs/images): signature, seal and handwriting
    # crops must survive so the middle.json can be re-rendered after correction.
    (out_dir / "middle.json").write_text(
        json.dumps(mj.to_dict(skip_defaults=True), ensure_ascii=False), encoding="utf-8"
    )
    content_list = render(mj, RenderFormat.CONTENT_LIST)
    (out_dir / "content_list.json").write_text(
        json.dumps(content_list, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    t1 = time.perf_counter()
    docx_bytes = render(mj, RenderFormat.DOCX)
    (out_dir / "mineru.docx").write_bytes(docx_bytes)
    render_s = time.perf_counter() - t1

    return {
        "file": src.name,
        "tier": tier,
        "pages": len(result.pages),
        "parse_s": round(parse_s, 2),
        "docx_render_s": round(render_s, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Files or directories")
    ap.add_argument("--tiers", nargs="+", default=["basic", "standard"])
    ap.add_argument("-o", "--out", default="Testcase/output")
    ap.add_argument("--warmup", action="store_true", help="Run each tier once before timing (excludes model load)")
    args = ap.parse_args()

    out_root = Path(args.out)
    files = collect_inputs(args.inputs)
    if not files:
        raise SystemExit("No input files found")

    timings = []
    for tier in args.tiers:
        if args.warmup:
            parse(files[0], tier=tier)
        for src in files:
            try:
                row = run_one(src, tier, out_root / src.stem / tier)
            except Exception as exc:  # keep going: one failing tier must not hide the others
                row = {"file": src.name, "tier": tier, "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            timings.append(row)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "timings.json").write_text(json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
