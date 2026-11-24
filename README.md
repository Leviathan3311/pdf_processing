# OCR → DOCX (MinerU)

Convert **scanned images and PDFs** (scanned, digital or mixed) into **editable Word documents that keep the
original layout**: fonts, sizes, bold / italic, colours, positions, real Word tables with merged cells, native
Word equations, and seals / signatures as transparent overlays. Built and tuned for **Vietnamese** administrative
documents.

```powershell
python scripts\convert.py Testcase -o D:\ket_qua
```

One command, batch-capable, writes only the `.docx` files.

---

## Highlights

| | |
|---|---|
| **Any input** | `jpg png webp tif bmp` and `pdf`; each PDF page is classified on its own (scan / text / formulas) |
| **Accurate OCR** | MinerU `standard` (VLM) on images flattened, deskewed and upscaled to 3200 px |
| **Vietnamese diacritics** | ProtonX T5 corrector, *constrained*: a change is kept only if the letters stay identical and only the accents differ; codes, numbers and names are locked |
| **Layout fidelity** | font size, weight, slant and colour measured on the image; a closed loop renders through Microsoft Word and moves every paragraph onto its original position |
| **Editable output** | text stays in the body flow (no text boxes), real `w:tbl` tables, OMML equations |
| **Digital PDFs** | text, fonts, tables and images read straight from the PDF: exact, no GPU, milliseconds per page |

## How it works

```
input (image / PDF)
 ├─ PDF page classifier (PyMuPDF) ── text page ─────────► PyMuPDF extractor ───────────────┐
 │                                   formulas ──► render ─┐                                │
 │                                   scan ──────► render ─┤                                │
 └─ image ───────────────────────────────────────────────┴► flatten · deskew · 3200 px    │
                                                             ► MinerU standard (middle.json)│
                                                             ► diacritic correction        │
                                                             ► layout measured on image ───┤
                                                                                           ▼
                                                          DOCX renderer (python-docx + OMML)
                                                                                           ▼
                                             closed loop: Word → PDF → compare baselines → fix
                                                                                           ▼
                                                                                      .docx
```

A batch shares every expensive start-up: MinerU loads once for all OCR pages while the spelling model loads in
parallel; spelling runs in one GPU pass; each closed-loop pass exports all documents with one Word instance.
A PDF with only text pages starts neither MinerU nor the spelling model.

## Requirements

- Windows 10/11, Python 3.11
- NVIDIA GPU recommended (tested: RTX 5070 Ti 16 GB); MinerU uses ONNX DirectML + llama.cpp
- Microsoft Word (for the closed loop; not needed with `--passes 0`). Word also provides `MML2OMML.XSL`,
  used to turn formulas into native equations

## Installation

```powershell
git clone -b minerU https://github.com/Leviathan3311/pdf_processing.git
cd pdf_processing
```

**1. Main environment**

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**2. MinerU** (separate venv, so its dependency pins never clash with the main environment)

```powershell
python -m venv .venv-mineru
.venv-mineru\Scripts\pip install -r requirements-mineru.txt
.venv-mineru\Scripts\mineru-models-download
```

**3. Spelling model** (`protonx-models/protonx-legal-tc`, ~0.9 GB)

```powershell
huggingface-cli download protonx-models/protonx-legal-tc --local-dir models/protonx-legal-tc
```

Model weights are not stored in git (`models/` is ignored).

## Usage

```powershell
$env:PYTHONUTF8 = "1"

# every image / PDF in a folder → D:\ket_qua\<name>.docx
python scripts\convert.py Testcase -o D:\ket_qua

# several files
python scripts\convert.py scan.jpg report.pdf -o D:\ket_qua

# one file → exactly this path
python scripts\convert.py cong_van.pdf -o D:\ket_qua\cong_van.docx
```

| Option | Default | Meaning |
|---|---|---|
| `-o, --out` | required | output folder, or a `.docx` path for a single input |
| `--passes` | `2` | closed-loop passes through Word: `2` most exact, `1` faster, `0` no Word |
| `--keep-work DIR` | off | keep intermediates and side-by-side `debug/*.compare.png` images |
| `--long-edge` | `3200` | OCR image size (px); 3200 measured best |
| `--model` | `models/protonx-legal-tc` | spelling model folder |
| `--mineru-python` | `.venv-mineru\Scripts\python.exe` | interpreter of the MinerU venv |

The run ends with a report per file: page kinds, diacritic fixes, Word page count, mean vertical offset from the
original (pt). A `<name>.gt.txt` next to an input adds its character error rate (CER) before / after correction.

## Results

Batch of 14 files / 17 pages (10 images, 4 scanned PDF pages, 3 digital PDF pages), `--passes 2`:
**all converted, 114 s total**, every document on its original page count.

| Input | Mean vertical offset |
|---|---|
| Digital PDF (Word export: letterhead, justified text, merged-cell table) | 0.1 pt |
| PDF with formulas (equations become native Word equations) | 0.1 pt |
| Clean scans and screenshots | 0.0 – 3.2 pt |
| Phone photos, faint or curled pages | 1.5 – 6 pt |
| Real 2-page scanned PDF with seal and signature | 8.9 pt |

Diacritic correction on a scanned official letter: word errors 18 → 8. Full measurements: [`PROGRESS.md`](PROGRESS.md).

## Project structure

```
src/conversion/
  pipeline.py           batch orchestration (input → OCR → spelling → layout → closed loop)
  page_classifier.py    per-page PDF kind: scan / text_complex / text_simple
  pymupdf_extractor.py  digital PDF pages → layout, straight from the text layer
  preprocess.py         background flattening, deskew, upscale; PDF page rendering
  spelling.py           constrained Vietnamese diacritic correction
  measure.py            typography measured on the image (lines, size, bold, italic, colour)
  layout.py             MinerU middle.json + image → positioned paragraphs, tables, floats (XY-cut)
  docx_renderer.py      layout → editable DOCX
  math_omml.py          LaTeX → MathML → Word OMML
  fit_check.py          closed-loop position correction
  word_export.py        DOCX → PDF through Word (one instance per batch)
scripts/
  convert.py            command-line entry point
  mineru_parse.py       MinerU runner (runs in .venv-mineru)
  render_layout.py      re-run only the layout step on a kept work folder
  eval_ocr.py           character error rate against hand-typed ground truth
  bench_llm.py          benchmark a spelling / LLM model
  spike_mineru.py       compare MinerU tiers
tests/                  unit tests (no models, no Word needed)
CLAUDE.md               design document (decisions, thresholds, rationale)
PROGRESS.md             measurements and status log
```

## Development

```powershell
python -m pytest tests -q                                      # unit tests
python scripts\convert.py Testcase -o out --keep-work work      # once
python scripts\render_layout.py work                            # iterate on layout without re-running OCR
python scripts\eval_ocr.py Testcase                             # CER (needs <name>.gt.txt)
```

Layout thresholds are always **relative to the page itself** (median gap, the page's ink colour, letter height
without diacritics, MinerU block type), never tuned to one sample; every change is re-run on the whole sample set.
Sample documents contain real personal data: keep them in `Testcase/` (git-ignored), never commit them.

## Limitations

- Bold / italic is measured per line, not per word: an italic phrase inside a paragraph is not reproduced.
- Handwriting is kept as an image; handwritten digits inside a printed line can enlarge that line.
- Digital PDFs: vector graphics other than thin rules (charts, diagrams) and rotated text are not yet carried over.
- PDF pages with formulas go through OCR (MinerU `flash` on the text layer is not wired yet).
- Windows only for now (Word COM for the closed loop, DirectML for MinerU).

## License

Code: [Apache 2.0](LICENSE). Third-party models keep their own licenses: MinerU — *MinerU Open Source License*;
`protonx-models/protonx-legal-tc` — **non-commercial** (for commercial use, swap in an Apache-2.0 corrector such as
`nrl-ai/vn-spell-correction-base` via `--model`).
