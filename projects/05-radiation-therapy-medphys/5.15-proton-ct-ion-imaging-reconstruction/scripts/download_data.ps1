# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL proton-CT data (Windows)
# ---------------------------------------------------------------------------
# Project 5.15 : Proton CT & Ion Imaging Reconstruction
#
# CONTRACT (CLAUDE.md §8): prints the source pointers; downloads NOTHING and
# NEVER bypasses credentials/registration. The committed synthetic sample is
# enough to run the demo offline; for a bigger synthetic problem use
# scripts/make_synthetic.py.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.15 -- Proton CT & Ion Imaging Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Real / standard proton-CT list-mode data:"
Write-Host "  * TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) or GATE -- simulate a"
Write-Host "    pCT scan and export per-proton entry/exit tracks + residual range,"
Write-Host "    then convert to this project's list-mode format (see data/README.md)."
Write-Host "  * PRaVDA / PRIMA proton-CT consortia -- prototype-scanner datasets"
Write-Host "    (verify current URLs; registration may be required -- this script"
Write-Host "    does NOT bypass it)."
Write-Host "  * ACE collaboration proton-CT phantom datasets (verify URL)."
Write-Host ""
Write-Host "Offline stand-in (no download, reproducible, SYNTHETIC):"
Write-Host "  python scripts/make_synthetic.py                       # the committed sample"
Write-Host "  python scripts/make_synthetic.py --n 48 --angles 90 --rays 48   # larger"
Write-Host ""
Write-Host "When wiring a real dataset, keep it idempotent:"
Write-Host "  1) skip the download if the file already exists with the right SHA256"
Write-Host "  2) print source URL + expected size + checksum"
Write-Host "  3) for credentialed sets, print registration instructions ONLY"
