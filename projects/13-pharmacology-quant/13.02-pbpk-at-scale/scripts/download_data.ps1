# ===========================================================================
# scripts/download_data.ps1  --  Realistic PBPK pointers (Windows)
# ---------------------------------------------------------------------------
# Project 13.02 : PBPK at Scale. Nothing to download.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 13.02 -- PBPK at Scale"
Write-Host ""
Write-Host "There is no file to download: the population is sampled from the"
Write-Host "parameters in data/sample/pbpk_params.txt."
Write-Host ""
Write-Host "For REAL whole-body PBPK (~15 compartments, literature physiology):"
Write-Host "  PK-Sim : https://github.com/Open-Systems-Pharmacology/PK-Sim"
Write-Host "  nvQSP  : https://github.com/NVIDIA-Digital-Bio/nvQSP   (GPU ODE solvers)"
Write-Host "  Open Systems Pharmacology suite: tissue volumes / blood flows databases."
Write-Host ""
Write-Host "Bigger population (no download):"
Write-Host "  python scripts/make_synthetic.py --patients 100000"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
