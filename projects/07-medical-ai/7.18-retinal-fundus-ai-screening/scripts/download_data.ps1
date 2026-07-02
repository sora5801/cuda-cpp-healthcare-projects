# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 7.18 : Retinal Fundus AI Screening
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. Every real fundus
# dataset here is account-gated, so this script prints instructions + links only
# and defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 7.18 -- Retinal Fundus AI Screening"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "All real fundus datasets are ACCESS-RESTRICTED. This script does NOT"
Write-Host "bypass any login or license -- it prints where to get them:"
Write-Host ""
Write-Host "  EyePACS   ~88,000 labelled fundus images, 5-grade DR severity."
Write-Host "            Kaggle 'Diabetic Retinopathy Detection' (account + rules). Verify URL."
Write-Host "  APTOS2019 3,662 fundus images, DR grading."
Write-Host "            Kaggle 'APTOS 2019 Blindness Detection' (account). Verify URL."
Write-Host "  DRIVE/STARE  retinal vessel-segmentation datasets (registration varies)."
Write-Host "  UK Biobank  ~68k fundus images + linked health records (credentialed):"
Write-Host "            https://www.ukbiobank.ac.uk/"
Write-Host ""
Write-Host "The committed tiny sample data/sample/fundus_sample.txt runs the demo offline."
Write-Host "To (re)generate a synthetic fundus image instead:"
Write-Host "    python scripts/make_synthetic.py --size 32"
Write-Host ""
Write-Host "To use a real image: load it (Pillow), resize to a small square, divide by"
Write-Host "255, and write it in the 'C H W label' + channel-major float format (data/README.md)."
