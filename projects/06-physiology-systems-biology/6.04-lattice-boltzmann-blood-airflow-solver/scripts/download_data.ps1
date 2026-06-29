# ===========================================================================
# scripts/download_data.ps1  --  Realistic LBM geometry pointers (Windows)
# ---------------------------------------------------------------------------
# Project 6.04 : Lattice-Boltzmann Blood/Airflow Solver
# This project generates its own flow from parameters; nothing to download.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 6.04 -- Lattice-Boltzmann Blood/Airflow Solver"
Write-Host ""
Write-Host "There is no file to download: the solver makes its own flow from the"
Write-Host "parameters in data/sample/channel_params.txt."
Write-Host ""
Write-Host "For REAL 3-D geometry (segmented vessels/airways + D3Q19/D3Q27):"
Write-Host "  HemeLB     : https://github.com/hemelb-codes/hemelb"
Write-Host "  PALABOS    : https://gitlab.com/unigespc/palabos"
Write-Host "  USERMESO-2 : https://github.com/AnselGitAccount/USERMESO-2.0"
Write-Host ""
Write-Host "Bigger 2-D grid:"
Write-Host "  python scripts/make_synthetic.py --nx 128 --ny 64 --steps 20000"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
