# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.16 : Functional MRI Analysis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. Real
# fMRI is credentialed and/or huge, so this project ships a SYNTHETIC sample and
# this script only prints where to get real data + defers to make_synthetic.py.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.16 : Functional MRI Analysis"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project runs on a SYNTHETIC sample (data/sample/fmri_sample.txt),"
Write-Host "so no download is required for the demo. Real public fMRI sources:"
Write-Host "  * HCP        https://db.humanconnectome.org/   (registration required)"
Write-Host "  * OpenNeuro  https://openneuro.org/            (BIDS; many open datasets)"
Write-Host "  * ABIDE      http://fcon_1000.projects.nitrc.org/indi/abide/"
Write-Host "  * UK Biobank https://www.ukbiobank.ac.uk/      (application + approval)"
Write-Host ""
Write-Host "Respect every dataset license; credentialed sets are NOT redistributed here."
Write-Host ""
Write-Host "For a larger SYNTHETIC problem (no download, fully reproducible):"
Write-Host "    python scripts/make_synthetic.py --V 200 --T 240"
Write-Host ""
Write-Host "To wire a REAL dataset later, follow this idempotent pattern:"
Write-Host "    1) skip download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
