# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.18 -- Image-Based 3D Printing / Model Generation for Surgery
#
# CONTRACT (CLAUDE.md sec.8): idempotent, documented, prints the source URL +
# access notes, and NEVER bypasses credentials/registration. The real clinical
# CT collections used to build patient-specific surgical models require
# registration and/or forbid redistribution, so this script only prints pointers
# and defers to scripts/make_synthetic.py for the offline, exactly-verifiable
# stand-in that the demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.18 -- Image-Based 3D Printing / Model Generation for Surgery"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC sphere volume (data/sample/volume_sample.txt)"
Write-Host "so the demo runs offline and its result is analytically verifiable. The real"
Write-Host "clinical datasets below are optional and gated behind registration/license:"
Write-Host ""
Write-Host "  TCIA body CT collections      https://www.cancerimagingarchive.net/   (per-collection license)"
Write-Host "  OsteoArthritis Initiative     https://nda.nih.gov/oai/                 (registration required)"
Write-Host "  VerSe vertebral CT            https://github.com/anjany/verse          (open)"
Write-Host "  TotalSegmentator dataset      https://zenodo.org/record/6802614        (CC BY)"
Write-Host ""
Write-Host "We do NOT auto-download credentialed data. To use a real volume, download it"
Write-Host "yourself, resample to a regular grid, and write it in the text format in"
Write-Host "data/README.md (nx ny nz spacing origin iso, then the samples)."
Write-Host ""
Write-Host "For a larger SYNTHETIC volume that needs no download, run:"
Write-Host "    python scripts/make_synthetic.py --n 65 --radius 24"
