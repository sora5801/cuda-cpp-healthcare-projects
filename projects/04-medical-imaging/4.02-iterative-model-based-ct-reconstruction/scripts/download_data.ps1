# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.2 : Iterative / Model-Based CT Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# licensing, and NEVER bypasses credentials/registration. The real low-dose CT
# datasets for this project are all credentialed / non-redistributable, so this
# script prints instructions + links and defers to scripts/make_synthetic.py for
# the offline synthetic stand-in the demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.2 -- Iterative / Model-Based CT Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The real datasets for this project are CREDENTIALED and NOT redistributable."
Write-Host "This script does not (and must not) bypass their registration. To obtain them:"
Write-Host ""
Write-Host "  * 2016 AAPM Low-Dose CT Grand Challenge (paired low/normal-dose scans)"
Write-Host "      https://www.aapm.org/grandchallenge/lowdosect/   (register for access)"
Write-Host "  * Mayo Clinic Low-Dose CT  -- via TCIA (The Cancer Imaging Archive)"
Write-Host "  * LIDC-IDRI CT scans       -- via TCIA, under a data-use agreement"
Write-Host "      https://www.cancerimagingarchive.net/"
Write-Host ""
Write-Host "After downloading (with your own credentials), convert a scan's sinogram to"
Write-Host "the text format documented in data/README.md."
Write-Host ""
Write-Host "The committed synthetic sample (data/sample/sinogram_sample.txt) is enough to"
Write-Host "run the demo offline. For a larger SYNTHETIC problem, run:"
Write-Host "    python scripts/make_synthetic.py --angles 90 --det 127 --img 96 --iters 80"
