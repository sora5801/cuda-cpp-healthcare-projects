# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.25 : Liver & Kidney Perfusion Modeling
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. The
# committed synthetic sample already runs the demo; the real sources below feed a
# richer, calibrated lobule/nephron model (an exercise in the README).
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.25 -- Liver & Kidney Perfusion Modeling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC lobule config (data/sample/lobule.txt), so"
Write-Host "no download is required to run the demo. To build a physiologically calibrated"
Write-Host "model, curate these PUBLIC sources into the loader's field layout:"
Write-Host ""
Write-Host "  * Human Protein Atlas -- liver zonal enzyme expression (Vmax gradient):"
Write-Host "      https://www.proteinatlas.org   (CC BY-SA 3.0; browse per-enzyme liver data)"
Write-Host "  * HMDB -- liver metabolite concentrations (set C_in / Km scales):"
Write-Host "      https://hmdb.ca                 (free for academic use; see terms)"
Write-Host "  * Open Systems Pharmacology PBPK model library -- organ clearance params:"
Write-Host "      https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library  (GPLv2)"
Write-Host "  * PhysioNet -- renal function datasets (credentialed for some sets):"
Write-Host "      https://physionet.org           (register; this script will NOT bypass it)"
Write-Host ""
Write-Host "  For a larger SYNTHETIC problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --nsin 1048576"
Write-Host ""
Write-Host "  Idempotent pattern when wiring a real fetch:"
Write-Host "    1) skip download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
