$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Desktop = Join-Path $Root "STEP1_ScrapingData\desktop_tool"
Set-Location $Desktop
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Desktop tool ready. Run: .\.venv\Scripts\python.exe -m app.main"
