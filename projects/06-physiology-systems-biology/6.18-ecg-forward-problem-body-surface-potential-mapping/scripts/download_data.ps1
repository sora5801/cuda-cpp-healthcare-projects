# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.18 : ECG Forward Problem & Body-Surface Potential Mapping
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. The real datasets
# this project points at are either registration-gated or ship full 3-D torso
# meshes far too large to commit, so this script only prints guidance and defers
# to scripts/make_synthetic.py for the offline, clearly-synthetic stand-in the
# demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.18 -- ECG Forward Problem & Body-Surface Potential Mapping"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project runs on a tiny SYNTHETIC sample (data/sample/ecg_sample.txt),"
Write-Host "so no download is required to build, run, and verify the demo."
Write-Host ""
Write-Host "Real-world data sources (study these; most need registration or are large):"
Write-Host "  * PhysioNet ECG databases          https://physionet.org"
Write-Host "      -- recorded surface ECGs (credentialed for some sets)."
Write-Host "  * EDGAR body-surface potential DB   https://edgar.sci.utah.edu  (verify URL)"
Write-Host "      -- multi-lead body-surface potential maps + torso geometries."
Write-Host "  * Visible Human torso geometry      https://www.nlm.nih.gov/research/visible/visible_human.html"
Write-Host "      -- a realistic torso volume conductor mesh (license/registration)."
Write-Host "  * Cardioid (LLNL) ECG module        https://github.com/llnl/cardioid"
Write-Host "  * openCARP ECG lead calculation     https://git.opencarp.org/openCARP/openCARP"
Write-Host ""
Write-Host "For a larger SYNTHETIC problem (more electrodes/sources/frames), run:"
Write-Host "  python scripts/make_synthetic.py --L 64 --S 8 --T 500"
Write-Host ""
Write-Host "When wiring a real dataset later, keep this script idempotent:"
Write-Host "  1) skip download if the file already exists with the right checksum"
Write-Host "  2) print source URL + expected size + SHA256"
Write-Host "  3) for credentialed sets, print registration instructions ONLY (never bypass)"
