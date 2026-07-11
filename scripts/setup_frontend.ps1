$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Frontend = Join-Path $Root "STEP6_Dashboard\frontend"
Set-Location $Frontend
npm install
Write-Host "Frontend ready. Run: npm run dev"
