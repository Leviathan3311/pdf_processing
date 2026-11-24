"""Batch planning of the pipeline (no models, no Word)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conversion.pipeline import collect_jobs  # noqa: E402


def _touch(folder: Path, *names: str) -> None:
    for name in names:
        (folder / name).write_bytes(b"x")


def test_folder_input_maps_each_image_and_pdf_to_one_docx(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    _touch(src, "a.jpg", "b.PNG", "notes.txt", "a.gt.txt", "c.pdf")
    jobs = collect_jobs([src], tmp_path / "out")
    assert [j.out.name for j in jobs] == ["a.docx", "b.docx", "c.docx"]
    assert all(j.out.parent == tmp_path / "out" for j in jobs)


def test_same_stem_inputs_do_not_overwrite_each_other(tmp_path):
    _touch(tmp_path, "a.jpg", "a.pdf")
    jobs = collect_jobs([tmp_path], tmp_path / "out")
    assert sorted(j.out.name for j in jobs) == ["a_jpg.docx", "a_pdf.docx"]


def test_single_image_to_exact_docx_path(tmp_path):
    _touch(tmp_path, "scan.jpg")
    jobs = collect_jobs([tmp_path / "scan.jpg"], tmp_path / "ket_qua" / "cong_van.docx")
    assert len(jobs) == 1 and jobs[0].out == tmp_path / "ket_qua" / "cong_van.docx"


def test_docx_path_with_many_inputs_is_rejected(tmp_path):
    _touch(tmp_path, "a.jpg", "b.jpg")
    with pytest.raises(SystemExit):
        collect_jobs([tmp_path], tmp_path / "one.docx")
