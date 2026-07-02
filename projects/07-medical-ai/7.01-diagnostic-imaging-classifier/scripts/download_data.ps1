# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 7.1 : Diagnostic Imaging Classifier   (reduced-scope teaching CNN)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# access notes, and NEVER bypasses credentials/registration. The real datasets
# for this project all require registration and forbid casual redistribution, so
# this script only prints instructions + links and defers to make_synthetic.py
# for the offline stand-in that the demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 7.1 -- Diagnostic Imaging Classifier"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project ships a SYNTHETIC sample (data/sample/imaging_sample.txt)"
Write-Host "and needs no download to run the demo. The real datasets below are"
Write-Host "CREDENTIALED / license-restricted -- fetch them yourself after agreeing to"
Write-Host "their terms; this script will not bypass registration."
Write-Host ""
Write-Host "  MIMIC-CXR   (credentialed, PhysioNet DUA):"
Write-Host "    https://physionet.org/content/mimic-cxr/"
Write-Host "  CheXpert    (registration, research-use license):"
Write-Host "    https://stanfordmlgroup.github.io/competitions/chexpert/"
Write-Host "  LIDC-IDRI   (TCIA, confirm per-collection terms):"
Write-Host "    https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI"
Write-Host "  TCIA        (per-collection licenses):"
Write-Host "    https://www.cancerimagingarchive.net/"
Write-Host ""
Write-Host "  To (re)generate the offline synthetic sample the demo uses:"
Write-Host "    python scripts/make_synthetic.py"
Write-Host ""
Write-Host "  When wiring a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
