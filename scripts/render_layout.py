"""Re-run only the layout step on a kept work folder (no OCR, no spelling) — handy while tuning layout.

    python scripts/convert.py Testcase -o out --keep-work work     # once
    python scripts/render_layout.py work [--stems aec455ba]         # then as often as needed

Reads work/prep/<stem>.jpg + work/fixed/<stem>.json; writes work/debug/<stem>.docx|.pdf|.compare.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversion.docx_renderer import render_pages  # noqa: E402
from conversion.fit_check import (OVERFLOW_PENALTY, correct_positions, restore_positions,  # noqa: E402
                                  shrink_lines, snapshot_positions)
from conversion.layout import build_from_middle  # noqa: E402
from conversion.word_export import compare_image, docx_to_pdf_batch, pdf_page_count  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("--stems", nargs="*", help="Only these stems (prefix match)")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    work = Path(args.work)
    debug = work / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    docs = {}
    for fixed in sorted((work / "fixed").glob("*.json")):
        stem = fixed.stem
        if args.stems and not any(stem.startswith(s) for s in args.stems):
            continue
        scan = work / "prep" / f"{stem}.jpg"
        docs[stem] = (scan, build_from_middle(json.loads(fixed.read_text(encoding="utf-8")), scan,
                                              work / "assets" / stem))
    best = {}
    for _ in range(args.passes):
        pairs = []
        for stem, (_, pages) in docs.items():
            render_pages(pages, debug / f"{stem}.docx")
            pairs.append((debug / f"{stem}.docx", debug / f"{stem}.pdf"))
        snaps = {stem: snapshot_positions(pages) for stem, (_, pages) in docs.items()}
        docx_to_pdf_batch(pairs)
        for stem, (_, pages) in docs.items():
            err = correct_positions(pages, debug / f"{stem}.pdf")
            if pdf_page_count(debug / f"{stem}.pdf") > len(pages):     # spilled: tighten and retry
                shrink_lines(pages)
                err += OVERFLOW_PENALTY
            if stem not in best or err < best[stem][0]:
                best[stem] = (err, snaps[stem])
    pairs = []
    for stem, (_, pages) in docs.items():
        if stem in best:
            restore_positions(pages, best[stem][1])
        render_pages(pages, debug / f"{stem}.docx")
        pairs.append((debug / f"{stem}.docx", debug / f"{stem}.pdf"))
    docx_to_pdf_batch(pairs)
    for stem, (scan, _) in docs.items():
        compare_image(scan, debug / f"{stem}.pdf", debug / f"{stem}.compare.png")
        err = best.get(stem, (float("nan"),))[0] % OVERFLOW_PENALTY
        print(f"{stem[:30]}: pages={pdf_page_count(debug / f'{stem}.pdf')}, |Δy| {err:.1f} pt -> "
              f"{debug / (stem + '.compare.png')}")


if __name__ == "__main__":
    main()
