# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 5.14 : GPU-Accelerated Adaptive MR-Linac Workflow (reduced-scope)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real MR-Linac images are patient data and
# cannot be redistributed here, so this script prints where to obtain them and
# defers to scripts/make_synthetic.py for the offline synthetic stand-in the demo
# actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.14 -- GPU-Accelerated Adaptive MR-Linac Workflow"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/oart_case.txt) is SYNTHETIC and is all the"
Write-Host "demo needs. Real MR-Linac data is patient-derived and NOT redistributed here."
Write-Host ""
Write-Host "To work with real MR-guided radiotherapy images, obtain them yourself from:"
Write-Host "  * MR-Linac Consortium shared datasets  -> verify URL at mrlinac.org (access)"
Write-Host "  * TCIA MR-guided RT collections        -> https://www.cancerimagingarchive.net/"
Write-Host "                                            (per-collection license / DUA)"
Write-Host "  * AAPM MR-Linac Working Group test cases -> AAPM task-group pages"
Write-Host "  * MRI-only radiotherapy cohorts        -> per published-paper terms"
Write-Host ""
Write-Host "Respect every license; some require registration or a data-use agreement."
Write-Host "This script intentionally does NOT bypass any of that."
Write-Host ""
Write-Host "For a larger SYNTHETIC slice instead, run:"
Write-Host "    python scripts/make_synthetic.py --nx 64 --ny 64"
