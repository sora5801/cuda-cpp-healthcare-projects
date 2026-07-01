# ===========================================================================
# scripts/download_data.ps1  --  Real-data calibration pointers (Windows)
# ---------------------------------------------------------------------------
# Project 6.8 : Tumor Growth & Treatment-Response Modeling
#
# There is NOTHING to download to run this project: the simulation is built
# deterministically from data/sample/tumor_params.txt. This script only prints
# where REAL data would come from to calibrate a model, and never bypasses any
# registration or credentials (CLAUDE.md §8).
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.8 -- Tumor Growth & Treatment-Response Modeling"
Write-Host ""
Write-Host "There is no file to download: the tumor field is built from the"
Write-Host "parameters in data/sample/tumor_params.txt (see data/README.md)."
Write-Host ""
Write-Host "This is a TEACHING model. Real mathematical-oncology models calibrate"
Write-Host "the parameters (D, rho, alpha, beta) against imaging + omics:"
Write-Host "  TCGA (multi-omics + imaging) : https://portal.gdc.cancer.gov"
Write-Host "  TCIA (tumor imaging)         : https://www.cancerimagingarchive.net"
Write-Host "  PhysioNet (oncology series)  : https://physionet.org"
Write-Host "  Zenodo (sim datasets)        : search 'tumor growth simulation'"
Write-Host ""
Write-Host "Some of these require registration; obtain access through their own"
Write-Host "portals -- this script will not attempt to bypass any credentials."
Write-Host ""
Write-Host "Bigger / different SYNTHETIC runs (no download):"
Write-Host "  python scripts/make_synthetic.py --nx 256 --ny 256 --steps 800"
Write-Host "  python scripts/make_synthetic.py --dose 3 --n-fractions 10"
Write-Host ""
Write-Host "Target data dir: $DataDir"
