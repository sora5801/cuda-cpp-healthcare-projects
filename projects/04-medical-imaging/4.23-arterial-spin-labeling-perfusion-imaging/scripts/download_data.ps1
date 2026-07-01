# ===========================================================================
# scripts/download_data.ps1  --  Point at REAL ASL datasets (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.23 : Arterial Spin Labeling & Perfusion Imaging
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. The real ASL datasets below are credentialed or
# large, so this script only PRINTS where to get them and how; the committed
# synthetic sample (scripts/make_synthetic.py) is what the demo actually runs.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.23 -- Arterial Spin Labeling & Perfusion Imaging"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The demo runs on the committed SYNTHETIC sample (data/sample/asl_sample.txt),"
Write-Host "so no download is required. For real multi-delay ASL data, see:"
Write-Host ""
Write-Host "  1) OpenNeuro ASL datasets (open, BIDS-formatted; search 'ASL'):"
Write-Host "       https://openneuro.org/"
Write-Host "     Many are directly downloadable (no credentials). Pick a multi-PLD/"
Write-Host "     multi-delay pCASL dataset; use the perf/ delta-M series + the PLD list."
Write-Host ""
Write-Host "  2) HCP ASL (Human Connectome Project, requires free registration + DUA):"
Write-Host "       https://db.humanconnectome.org/"
Write-Host ""
Write-Host "  3) ISMRM 2015 ASL challenge data (community reconstruction challenge)."
Write-Host ""
Write-Host "  4) UK Biobank ASL pilot data (requires an approved UK Biobank application)."
Write-Host ""
Write-Host "For (2)-(4) this script intentionally does NOT attempt to bypass login/DUA."
Write-Host "Register through the portal, then convert one subject's multi-delay delta-M"
Write-Host "series into the loader format documented in data/README.md."
Write-Host ""
Write-Host "Bigger SYNTHETIC study (no download):"
Write-Host "  python scripts/make_synthetic.py --voxels 1000000"
