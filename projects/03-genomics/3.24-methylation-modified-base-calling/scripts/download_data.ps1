# ===========================================================================
# scripts/download_data.ps1  --  Real methylation-data pointers (Windows)
# ---------------------------------------------------------------------------
# Project 3.24 : Methylation / Modified-Base Calling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This project SHIPS SYNTHETIC DATA
# (data/sample/) and needs no download to run the demo; this script only points
# at real datasets for further study and defers to make_synthetic.py for scale.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 3.24 -- Methylation / Modified-Base Calling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Nothing to download: the committed synthetic sample in data/sample/ runs the demo."
Write-Host ""
Write-Host "Real data for further study (basecall + event-align first; see data/README.md):"
Write-Host "  ONT open datasets (R10.4.1, 5mC/6mA labels) : https://github.com/GoekeLab/awesome-nanopore"
Write-Host "  ENCODE WGBS (ground-truth methylation)      : https://www.encodeproject.org/"
Write-Host "  NCBI GEO methylation studies                : https://www.ncbi.nlm.nih.gov/geo/"
Write-Host ""
Write-Host "Tools that produce the per-site event windows this project consumes:"
Write-Host "  f5c    (CUDA event alignment + meth calling) : https://github.com/hasindu2008/f5c"
Write-Host "  Dorado (basecalling + mod calling)           : https://github.com/nanoporetech/dorado"
Write-Host "  Remora (modified-base models)                : https://github.com/nanoporetech/remora"
Write-Host ""
Write-Host "Bigger synthetic instance (no download):"
Write-Host "  python scripts/make_synthetic.py --sites 4096"
