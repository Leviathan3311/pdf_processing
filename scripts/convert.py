"""Images and PDFs (scanned, text or mixed) → layout-preserving, spell-checked DOCX.
Only the .docx files are written; one .docx per input file, one Word page per input page.

    python scripts/convert.py Testcase -o D:/ket_qua               # batch: every image / PDF → D:/ket_qua/<stem>.docx
    python scripts/convert.py a.jpg b.pdf -o D:/ket_qua            # several files
    python scripts/convert.py scan.jpg -o D:/ket_qua/cong_van.docx # one file → exactly this path
    python scripts/convert.py Testcase -o out --passes 1           # faster, slightly less exact positions
    python scripts/convert.py Testcase -o out --keep-work work     # keep intermediates + compare images

Needs .venv-mineru (MinerU 4.x) and models/protonx-legal-tc for scanned pages (a PDF with only text
pages needs neither), and Microsoft Word for --passes > 0. Set PYTHONUTF8=1 on Windows.
A <stem>.gt.txt next to an input adds its character error rate to the report.
Handwritten blocks are read again by Qwen3.5-9B (tools/llama.cpp/llama-server.exe +
models/qwen3.5-9b/*.gguf) when installed; otherwise MinerU's text is kept (--hw-ocr off forces that).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversion.pipeline import (DEFAULT_HW_MMPROJ, DEFAULT_HW_MODEL, DEFAULT_HW_SERVER,  # noqa: E402
                                 DEFAULT_MINERU_PYTHON, DEFAULT_MODEL, HwOptions, convert, report)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Image / PDF files and/or folders of them")
    ap.add_argument("-o", "--out", required=True, help="Output folder, or a .docx path for a single input")
    ap.add_argument("--passes", type=int, default=2,
                    help="Closed-loop passes through Word (0 = no Word, fastest; 2 = most exact)")
    ap.add_argument("--keep-work", default=None, help="Keep intermediates in this folder (debugging)")
    ap.add_argument("--long-edge", type=int, default=3200)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--mineru-python", default=str(DEFAULT_MINERU_PYTHON))
    ap.add_argument("--hw-ocr", choices=["auto", "off", "force"], default="auto",
                    help="Handwriting re-read by a local VLM: auto = handwritten blocks only, off = MinerU only, "
                         "force = every prose block (debugging)")
    ap.add_argument("--hw-server", default=str(DEFAULT_HW_SERVER), help="llama-server executable")
    ap.add_argument("--hw-model", default=str(DEFAULT_HW_MODEL), help="VLM GGUF")
    ap.add_argument("--hw-mmproj", default=str(DEFAULT_HW_MMPROJ), help="vision projector GGUF")
    ap.add_argument("--hw-slots", type=int, default=4, help="parallel VLM requests")
    args = ap.parse_args()

    jobs = convert([Path(p) for p in args.inputs], Path(args.out), args.passes, args.long_edge, Path(args.model),
                   Path(args.mineru_python), Path(args.keep_work) if args.keep_work else None,
                   hw=HwOptions(args.hw_ocr, Path(args.hw_server), Path(args.hw_model), Path(args.hw_mmproj),
                                args.hw_slots))
    print()
    print(report(jobs))
    sys.exit(1 if any(j.error for j in jobs) else 0)


if __name__ == "__main__":
    main()
