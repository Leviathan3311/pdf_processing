"""Images and PDFs → layout-preserving, spell-checked DOCX, in batches (CLAUDE.md section 3).

    python scripts/convert.py Testcase -o out/            # every image / PDF in Testcase → out/<stem>.docx
    python scripts/convert.py scan.pdf -o result.docx     # one file → exactly that path

Every PDF page is classified on its own (page_classifier):
  - scan pages (and image files) → flatten, deskew, upscale → MinerU standard → diacritic fix → measured layout
  - text pages with formulas      → rendered → MinerU (formulas as LaTeX → Word equations)
  - other text pages              → read straight from the PDF (exact text, fonts, tables; no OCR, no GPU)
One DOCX per input file, one Word page per input page.

Only the .docx files are written. Intermediates live in a temporary folder that is deleted at the end
(`work_dir` keeps them for debugging: prep/, ocr/, fixed/, debug/<stem>[__pNNN].compare.png).

A batch shares every expensive start-up:
  - MinerU (own venv) loads its models once for all OCR pages; the spelling model loads meanwhile
    (neither starts when no page needs OCR)
  - the spelling model corrects all documents in one batched GPU pass
  - each closed-loop pass exports ALL documents with ONE Microsoft Word instance
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .docx_renderer import render_pages
from .fit_check import (OVERFLOW_PENALTY, correct_positions, restore_positions, shrink_lines,
                        snapshot_positions)
from .layout import build_from_middle
from .word_export import compare_image, docx_to_pdf_batch, pdf_page_count

ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
INPUT_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}
DEFAULT_MODEL = ROOT / "models" / "protonx-legal-tc"
DEFAULT_MINERU_PYTHON = ROOT / ".venv-mineru" / "Scripts" / "python.exe"


@dataclass
class Unit:
    """One page of an input: OCR'd (image file, scanned or formula PDF page) or read from the PDF text layer."""
    key: str                                   # file stem in the work folder
    kind: str                                  # image | scan | text_complex | text_simple
    page_pt: tuple[float, float] | None = None  # real page size (PDF pages)
    page: object | None = None                 # layout.Page, built directly for text_simple
    raw: dict | None = None                    # MinerU middle.json
    fixed: dict | None = None                  # after the diacritic fix

    @property
    def ocr(self) -> bool:
        return self.page is None


@dataclass
class Job:
    src: Path
    out: Path
    spelling_changes: int = 0
    layout_dy: float | None = None        # mean |Δy| (pt) between source lines and Word lines, best pass
    word_pages: int | None = None
    cer_before: float | None = None
    cer_after: float | None = None
    error: str | None = None
    units: list[Unit] = field(default_factory=list, repr=False)
    _pages: list | None = field(default=None, repr=False)

    @property
    def key(self) -> str:                 # unique name inside the work folder
        return self.out.stem

    @property
    def kinds(self) -> str:
        counts = Counter(u.kind for u in self.units)
        return ", ".join(f"{n} {k}" for k, n in counts.items())


def collect_jobs(inputs: list[Path], out: Path, log=print) -> list[Job]:
    """Inputs (files and/or folders) → jobs. `out` is a folder, or a .docx path for a single input."""
    files: list[Path] = []
    for p in map(Path, inputs):
        files += sorted(f for f in p.iterdir() if f.is_file()) if p.is_dir() else [p]
    sources = [f for f in files if f.suffix.lower() in INPUT_SUFFIXES]
    if out.suffix.lower() == ".docx":
        if len(sources) != 1:
            raise SystemExit("A .docx output path needs exactly one input file; give a folder for batches")
        return [Job(sources[0], out)]
    jobs = [Job(f, out / f"{f.stem}.docx") for f in sources]
    seen = Counter(j.key for j in jobs)                 # a.pdf and a.jpg must not overwrite each other
    for j in jobs:
        if seen[j.key] > 1:
            j.out = out / f"{j.src.stem}_{j.src.suffix.lstrip('.').lower()}.docx"
    return jobs


def _merged(units: list[Unit], attr: str) -> dict:
    return {"pages": [p for u in units if getattr(u, attr) for p in getattr(u, attr).get("pages", [])]}


def _score(job: Job) -> None:
    gt = job.src.with_suffix(".gt.txt")
    ocr = [u for u in job.units if u.ocr and u.raw is not None]
    if not gt.exists() or not ocr:
        return
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_ocr import cer, middle_text, normalise
    truth = normalise(gt.read_text(encoding="utf-8"))
    job.cer_before = cer(normalise(middle_text(_merged(ocr, "raw"))), truth)
    job.cer_after = cer(normalise(middle_text(_merged(ocr, "fixed"))), truth)


def convert(inputs: list[Path], out: Path, passes: int = 2, long_edge: int = 3200, model: Path = DEFAULT_MODEL,
            mineru_python: Path = DEFAULT_MINERU_PYTHON, work_dir: Path | None = None, log=print) -> list[Job]:
    jobs = collect_jobs(inputs, Path(out), log)
    if not jobs:
        raise SystemExit("No input images or PDFs")
    keep = work_dir is not None
    tmp = None if keep else tempfile.TemporaryDirectory(prefix="ocr_docx_")
    work = Path(work_dir) if keep else Path(tmp.name)
    try:
        return _run(jobs, work, passes, long_edge, model, mineru_python, keep, log)
    finally:
        if tmp is not None:
            tmp.cleanup()


def _prepare_inputs(jobs: list[Job], prep: Path, work: Path, long_edge: int) -> None:
    """Fill each job's units: classify PDF pages, extract text pages, write OCR page images to `prep`."""
    import fitz  # PyMuPDF

    from .page_classifier import PageKind, classify_page
    from .preprocess import prepare, render_pdf_page
    from .pymupdf_extractor import extract_page
    for job in jobs:
        try:
            if job.src.suffix.lower() != ".pdf":
                job.units = [Unit(job.key, "image")]
                prepare(job.src, prep / f"{job.key}.jpg", long_edge)
                continue
            with fitz.open(str(job.src)) as doc:
                if doc.needs_pass:
                    raise ValueError("password-protected PDF")
                if doc.page_count == 0:
                    raise ValueError("PDF has no pages")
                for i, page in enumerate(doc):
                    kind, _ = classify_page(page)
                    unit = Unit(f"{job.key}__p{i + 1:03d}", kind.value, (page.rect.width, page.rect.height))
                    if kind is PageKind.TEXT_SIMPLE:
                        unit.page = extract_page(page, work / "assets" / unit.key)
                    else:
                        raw = prep / f"{unit.key}.png"
                        render_pdf_page(page, raw, long_edge, scan=kind is PageKind.SCAN)
                        prepare(raw, prep / f"{unit.key}.jpg", long_edge, clean=kind is not PageKind.SCAN)
                        raw.unlink()
                    job.units.append(unit)
        except Exception as exc:                       # one bad file must not stop the batch
            job.error = f"input: {type(exc).__name__}: {exc}"
            job.units = []


def _run(jobs, work: Path, passes, long_edge, model, mineru_python, keep, log) -> list[Job]:
    prep, ocr, docx_dir = work / "prep", work / "ocr", work / "docx"
    for d in (prep, ocr, docx_dir):
        d.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    t = time.perf_counter()
    _prepare_inputs(jobs, prep, work, long_edge)
    units = [u for j in jobs for u in j.units]
    ocr_units = [u for u in units if u.ocr]
    kinds = Counter(u.kind for u in units)
    log(f"[1/5] input     {len(jobs)} file(s), {len(units)} page(s): "
        f"{', '.join(f'{n} {k}' for k, n in kinds.items()) or 'none'}   {time.perf_counter() - t:5.1f} s")

    # MinerU runs in its own process while this one loads the spelling model; skipped when no page needs OCR
    t = time.perf_counter()
    corrector = None
    if ocr_units:
        mineru = subprocess.Popen([str(mineru_python), str(ROOT / "scripts" / "mineru_parse.py"), str(prep),
                                   "-o", str(ocr)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                  encoding="utf-8")
        from .spelling import Corrector
        corrector = Corrector(model)
        mineru_log, _ = mineru.communicate()
        for job in jobs:
            for u in job.units:
                path = ocr / f"{u.key}.json"
                if not u.ocr:
                    continue
                if path.exists():
                    u.raw = json.loads(path.read_text(encoding="utf-8"))
                elif not job.error:
                    job.error = "OCR failed: " + next((l for l in mineru_log.splitlines() if u.key in l),
                                                      f"{u.key}: no output")
    log(f"[2/5] ocr       MinerU standard on {len(ocr_units)} page(s) (+ spelling model load)"
        f"  {time.perf_counter() - t:5.1f} s")

    t = time.perf_counter()
    read = [u for u in ocr_units if u.raw is not None]
    for u in read:
        u.fixed = copy.deepcopy(u.raw)
    if corrector is not None and read:
        from .spelling import correct_many
        changes = {id(u): len(r.changes) for u, r in zip(read, correct_many([u.fixed for u in read], corrector))}
        for job in jobs:
            job.spelling_changes = sum(changes.get(id(u), 0) for u in job.units)
    del corrector
    log(f"[3/5] spelling  {sum(j.spelling_changes for j in jobs)} diacritic fix(es)                "
        f"{time.perf_counter() - t:5.1f} s")

    t = time.perf_counter()
    for job in jobs:
        if job.error or not job.units:
            continue
        try:
            pages = []
            for u in job.units:
                pages += [u.page] if u.page is not None else build_from_middle(
                    u.fixed, prep / f"{u.key}.jpg", work / "assets" / u.key, u.page_pt)
            job._pages = pages
        except Exception as exc:                       # one bad document must not stop the batch
            job.error = f"layout: {type(exc).__name__}: {exc}"
    laid = [j for j in jobs if j._pages]
    log(f"[4/5] layout    measure + typeset                       {time.perf_counter() - t:5.1f} s")

    # closed loop: every pass renders all documents, one Word instance exports all of them,
    # each paragraph moves by its own measured error; keep each document's best pass
    t = time.perf_counter()
    best: dict[str, tuple] = {}
    for _ in range(passes):
        pairs = []
        for job in laid:
            render_pages(job._pages, docx_dir / f"{job.key}.docx")
            pairs.append((docx_dir / f"{job.key}.docx", docx_dir / f"{job.key}.pdf"))
        snapshots = {job.key: snapshot_positions(job._pages) for job in laid}
        docx_to_pdf_batch(pairs)
        for job in laid:
            pdf = docx_dir / f"{job.key}.pdf"
            err = correct_positions(job._pages, pdf)
            n_pages = pdf_page_count(pdf)
            if n_pages > len(job._pages):                  # spilled onto an extra page: tighten, retry
                shrink_lines(job._pages)
                err += OVERFLOW_PENALTY
            if job.key not in best or err < best[job.key][0]:
                best[job.key] = (err, snapshots[job.key], n_pages)
    # rescue: documents whose best render still spills onto an extra page are tightened a little more
    # until they fit (only these documents go through Word again)
    for attempt in range(3):
        spilled = [j for j in laid if j.key in best and best[j.key][2] > len(j._pages)]
        if not spilled:
            break
        pairs = []
        for job in spilled:
            restore_positions(job._pages, best[job.key][1])
            shrink_lines(job._pages, 0.95 ** (attempt + 1))
            render_pages(job._pages, docx_dir / f"{job.key}.docx")
            pairs.append((docx_dir / f"{job.key}.docx", docx_dir / f"{job.key}.pdf"))
        snapshots = {job.key: snapshot_positions(job._pages) for job in spilled}
        docx_to_pdf_batch(pairs)
        for job in spilled:
            pdf = docx_dir / f"{job.key}.pdf"
            n_pages = pdf_page_count(pdf)
            err = correct_positions(job._pages, pdf) + (OVERFLOW_PENALTY if n_pages > len(job._pages) else 0)
            if err < best[job.key][0]:
                best[job.key] = (err, snapshots[job.key], n_pages)
    for job in laid:
        if job.key in best:
            err, snapshot, pages = best[job.key]
            restore_positions(job._pages, snapshot)
            job.layout_dy, job.word_pages = round(err % OVERFLOW_PENALTY, 2), pages
        job.out.parent.mkdir(parents=True, exist_ok=True)
        render_pages(job._pages, job.out)
        _score(job)
    log(f"[5/5] word      {passes} closed-loop pass(es), 1 Word start each   {time.perf_counter() - t:5.1f} s")

    if keep:                                           # debugging aids only when the work folder is kept
        _debug(jobs, laid, work, prep)

    log(f"done: {sum(1 for j in jobs if not j.error and j.out.exists())}/{len(jobs)} document(s) "
        f"in {time.perf_counter() - t_all:.1f} s")
    return jobs


def _debug(jobs, laid, work: Path, prep: Path) -> None:
    import fitz  # PyMuPDF
    debug, fixed = work / "debug", work / "fixed"
    debug.mkdir(exist_ok=True)
    fixed.mkdir(exist_ok=True)
    for u in (u for j in jobs for u in j.units if u.fixed is not None):
        (fixed / f"{u.key}.json").write_text(json.dumps(u.fixed, ensure_ascii=False), encoding="utf-8")
    pairs = [(job.out, debug / f"{job.key}.pdf") for job in laid]
    docx_to_pdf_batch(pairs)
    for job, (_, pdf) in zip(laid, pairs):
        for n, u in enumerate(job.units):
            src = prep / f"{u.key}.jpg"
            if not src.exists():                       # text page: picture of the original PDF page
                src = debug / f"{u.key}.src.png"
                with fitz.open(str(job.src)) as doc:
                    doc[n].get_pixmap(dpi=150).save(str(src))
            compare_image(src, pdf, debug / f"{u.key}.compare.png", page_no=n)


def report(jobs: list[Job]) -> str:
    def pct(v):
        return "—" if v is None else f"{v:.2%}"
    rows = [f"{'input':<34} {'pages':<26} {'fixes':>5} {'word':>4} {'|Δy| pt':>8} {'CER before→after':>18}"
            f"  output / error"]
    for j in jobs:
        dy = "—" if j.layout_dy is None else f"{j.layout_dy:.1f}"
        rows.append(f"{j.src.name[:34]:<34} {j.kinds[:26]:<26} {j.spelling_changes:>5} {j.word_pages or '—':>4} "
                    f"{dy:>8} {pct(j.cer_before) + ' → ' + pct(j.cer_after):>18}  {j.error or j.out}")
    return "\n".join(rows)
