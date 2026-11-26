"""Download every model and tool the pipeline needs that is not in git (weights are too large for GitHub).

    python scripts\\download_models.py                 # all: spelling model, handwriting VLM, llama.cpp
    python scripts\\download_models.py --only spelling  # just one part
    python scripts\\download_models.py --quant Q4_K_M   # smaller VLM (7.4 instead of 8.8 GiB VRAM)

Files already present are skipped, so the script can be run again after an interrupted download.
MinerU's own models are downloaded by `.venv-mineru\\Scripts\\mineru-models-download` (see setup.ps1).

| Part        | Source                                              | Goes to                     | Size    |
|-------------|-----------------------------------------------------|-----------------------------|---------|
| spelling    | huggingface.co/protonx-models/protonx-legal-tc      | models/protonx-legal-tc     | ~0.9 GB |
| handwriting | huggingface.co/unsloth/Qwen3.5-9B-GGUF (+ mmproj)   | models/qwen3.5-9b           | ~7.9 GB |
| llama       | github.com/ggml-org/llama.cpp release b12777 (CUDA) | tools/llama.cpp             | ~0.8 GB |
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPELLING_REPO = "protonx-models/protonx-legal-tc"
VLM_REPO = "unsloth/Qwen3.5-9B-GGUF"
VLM_MMPROJ = "mmproj-F16.gguf"                   # keep the vision part in f16: quantising it hurts OCR most
LLAMA_TAG = "b12777"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"
LLAMA_BUILDS = {                                 # --llama-build → release assets (CUDA runtime zip when needed)
    "cuda-13.5": [f"llama-{LLAMA_TAG}-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip"],
    "cuda-12.4": [f"llama-{LLAMA_TAG}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"],
    "vulkan": [f"llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip"],
    "cpu": [f"llama-{LLAMA_TAG}-bin-win-cpu-x64.zip"],
}


def spelling() -> None:
    from huggingface_hub import snapshot_download
    out = ROOT / "models" / "protonx-legal-tc"
    print(f"spelling model {SPELLING_REPO} → {out}")
    snapshot_download(SPELLING_REPO, local_dir=out)


def handwriting(quant: str) -> None:
    from huggingface_hub import hf_hub_download
    out = ROOT / "models" / "qwen3.5-9b"
    for name in (f"Qwen3.5-9B-{quant}.gguf", VLM_MMPROJ):
        if (out / name).exists():
            print(f"handwriting VLM {name}: already there")
            continue
        print(f"handwriting VLM {VLM_REPO}/{name} → {out}")
        hf_hub_download(VLM_REPO, name, local_dir=out)
    if quant != "Q6_K":
        print(f"  note: the pipeline defaults to Q6_K; run convert.py with --hw-model {out / f'Qwen3.5-9B-{quant}.gguf'}")


def llama(build: str) -> None:
    out = ROOT / "tools" / "llama.cpp"
    if (out / "llama-server.exe").exists():
        print(f"llama.cpp: already there ({out})")
        return
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name in LLAMA_BUILDS[build]:
            url = LLAMA_URL.format(tag=LLAMA_TAG, name=name)
            zpath = Path(tmp) / name
            print(f"llama.cpp {url}")
            with urllib.request.urlopen(url) as r, open(zpath, "wb") as f:
                shutil.copyfileobj(r, f)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(out)
    print(f"llama.cpp → {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=["spelling", "handwriting", "llama"],
                    default=["spelling", "handwriting", "llama"])
    ap.add_argument("--quant", default="Q6_K", choices=["Q4_K_M", "Q6_K", "Q8_0"],
                    help="handwriting VLM size: Q6_K (default, 8.8 GiB VRAM), Q4_K_M (7.4), Q8_0 (10.5)")
    ap.add_argument("--llama-build", default="cuda-13.5", choices=sorted(LLAMA_BUILDS),
                    help="llama.cpp build: cuda-13.5 (RTX 50xx, recent driver), cuda-12.4 (older driver), vulkan, cpu")
    args = ap.parse_args()
    if "spelling" in args.only:
        spelling()
    if "handwriting" in args.only:
        handwriting(args.quant)
    if "llama" in args.only:
        llama(args.llama_build)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
