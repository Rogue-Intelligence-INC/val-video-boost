# Valhalla Fate consistency stack -- one-shot Windows install.
# Creates .venv in the repo root, installs CUDA torch + requirements,
# then runs the environment doctor. Model weights are NOT downloaded
# by this script (see docs/LAUNCH.md for the model list); point
# VAL_MODELS_ROOT at an existing model library to skip downloads.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"

Write-Host "== val-video-boost install ==" -ForegroundColor Cyan
Write-Host "repo: $Root"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not found on PATH. Install Python 3.11+ first."
}
python -c "import sys; assert sys.version_info >= (3,10), sys.version" 

if (-not (Test-Path $Venv)) {
    Write-Host "creating venv..."
    python -m venv $Venv
}
$Py = Join-Path $Venv "Scripts\python.exe"

Write-Host "installing torch (CUDA 12.8 index)..."
& $Py -m pip install --upgrade pip -q
& $Py -m pip install torch --index-url https://download.pytorch.org/whl/cu128

Write-Host "installing requirements..."
& $Py -m pip install -r (Join-Path $Root "requirements.txt")

if (-not $env:VAL_MODELS_ROOT) {
    Write-Host "VAL_MODELS_ROOT not set - defaulting to C:\Ai\models (set it to your model library)" -ForegroundColor Yellow
}

Write-Host "running doctor..."
& $Py (Join-Path $Root "code\doctor.py")
Write-Host ""
Write-Host "Next:  cd code; ..\.venv\Scripts\python.exe run_pipeline.py" -ForegroundColor Green
Write-Host "API:   ..\.venv\Scripts\python.exe fate_service.py   (http://127.0.0.1:8790/health)"
