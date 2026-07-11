$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "STEP6_Dashboard\backend"
Set-Location $Backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Backend ready. Run: .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload"
