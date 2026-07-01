# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 5.7 : Brachytherapy Dose & Source Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL, and
# NEVER bypasses credentials/registration. The real TG-43 consensus source
# datasets live in published journal tables (below); we do not redistribute
# them. The committed synthetic sample runs the demo offline; this script only
# prints where to obtain the real datasets.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.7 : Brachytherapy Dose & Source Modeling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC plan in data/sample/plan_sample.txt;"
Write-Host "no download is required to run the demo. Real TG-43 datasets:"
Write-Host ""
Write-Host "  * AAPM TG-43U1 consensus source data (radial dose g_L(r) and"
Write-Host "    anisotropy F(r,theta) tables per source model, e.g. Ir-192 HDR,"
Write-Host "    Pd-103, I-125): https://www.aapm.org/pubs/reports/"
Write-Host "  * ESTRO ACROP brachytherapy guideline test cases (planning geometry)."
Write-Host "  * TCIA prostate brachytherapy CT datasets (imaging; free registration):"
Write-Host "    https://www.cancerimagingarchive.net/"
Write-Host ""
Write-Host "To transcribe a real source's tables into this project's plan format,"
Write-Host "edit data/sample/plan_sample.txt (format documented in data/README.md)."
Write-Host "For a larger SYNTHETIC grid, run:"
Write-Host "    python scripts/make_synthetic.py --grid 81"
