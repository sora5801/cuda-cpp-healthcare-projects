# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.27 : Parameter Estimation & Data Assimilation for Physiological Models
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The clinical waveform datasets below all
# require registration/credentialed access, so this script only prints
# instructions + links and defers to scripts/make_synthetic.py for the offline,
# fully-reproducible synthetic stand-in the demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.27 -- Parameter Estimation & Data Assimilation"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The demo runs entirely on the committed SYNTHETIC sample (data/sample/enkf_config.txt)."
Write-Host "Real clinical waveform / cardiac-parameter datasets (all require registration):"
Write-Host "  * PhysioNet MIMIC clinical waveforms  https://physionet.org  (credentialed)"
Write-Host "  * UK Biobank cardiac functional params https://www.ukbiobank.ac.uk  (application)"
Write-Host "  * Zenodo cardiac mechanics emulation   https://zenodo.org/records/7075055"
Write-Host "  * openCARP community experiments       https://opencarp.org/community/community-experiments"
Write-Host ""
Write-Host "This script does NOT attempt to bypass any credential wall (CLAUDE.md 8)."
Write-Host "For a larger SYNTHETIC problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --ensemble 1024 --windows 80"
