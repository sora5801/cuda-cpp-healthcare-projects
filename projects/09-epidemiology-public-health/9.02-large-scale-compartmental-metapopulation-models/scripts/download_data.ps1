# ===========================================================================
# scripts/download_data.ps1  --  Realistic epidemic-model pointers (Windows)
# ---------------------------------------------------------------------------
# Project 9.02 : Large-Scale Compartmental & Metapopulation Models. Nothing to fetch.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 9.02 -- Large-Scale Compartmental & Metapopulation Models"
Write-Host ""
Write-Host "There is no file to download: the program derives every member's"
Write-Host "parameters from the sweep in data/sample/ensemble_params.txt."
Write-Host ""
Write-Host "For REAL models (mobility matrices, age structure, many patches):"
Write-Host "  MEmilio     : https://github.com/SciCompMod/memilio   (C++/CUDA)"
Write-Host "  EpiModel    : https://github.com/EpiModel/EpiModel    (R, network)"
Write-Host "  Torchdiffeq : https://github.com/rtqichen/torchdiffeq (GPU ODE solvers)"
Write-Host ""
Write-Host "Bigger ensemble (no download):"
Write-Host "  python scripts/make_synthetic.py --nb 200 --ng 200"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
