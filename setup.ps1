# One-command install on Windows: both venvs, their packages, and every model (~10 GB download).
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Quant Q4_K_M -LlamaBuild cuda-12.4
# Safe to run again: existing venvs and downloaded files are reused.
param(
    [string]$Python = "python",
    [ValidateSet("Q4_K_M", "Q6_K", "Q8_0")][string]$Quant = "Q6_K",
    [ValidateSet("cuda-13.5", "cuda-12.4", "vulkan", "cpu")][string]$LlamaBuild = "cuda-13.5"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"

function Invoke-Step([string]$Title, [scriptblock]$Body) {
    Write-Host "`n=== $Title" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) { throw "$Title failed (exit $LASTEXITCODE)" }
}

Invoke-Step "1/4 main venv (.venv) + requirements.txt" {
    if (-not (Test-Path .venv\Scripts\python.exe)) { & $Python -m venv .venv }
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt
}
Invoke-Step "2/4 MinerU venv (.venv-mineru) + requirements-mineru.txt" {
    if (-not (Test-Path .venv-mineru\Scripts\python.exe)) { & $Python -m venv .venv-mineru }
    .venv-mineru\Scripts\python -m pip install --upgrade pip
    .venv-mineru\Scripts\python -m pip install -r requirements-mineru.txt
}
Invoke-Step "3/4 MinerU models" {
    .venv-mineru\Scripts\mineru-models-download
}
Invoke-Step "4/4 spelling model, handwriting VLM, llama.cpp" {
    .venv\Scripts\python scripts\download_models.py --quant $Quant --llama-build $LlamaBuild
}
Write-Host "`nReady. Try: .venv\Scripts\python scripts\convert.py <folder> -o <out>" -ForegroundColor Green
