# ===========================================================================
# scripts/download_data.ps1  --  Realistic PK/PD & PBPK pointers (Windows)
# ---------------------------------------------------------------------------
# Project 6.15 : PK/PD & PBPK Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. There is NOTHING to download for this project: the
# virtual population is sampled from the parameters in data/sample/pkpd_params.txt.
# This script only prints where to get REAL PK/PD data and models (some of which
# require registration -- we link and instruct, never scrape).
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.15 -- PK/PD & PBPK Modeling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "There is no file to download: the virtual population is generated from"
Write-Host "the parameters in data/sample/pkpd_params.txt (see scripts/make_synthetic.py)."
Write-Host ""
Write-Host "For REAL clinical PK data and validated PK/PD & PBPK models:"
Write-Host "  PhysioNet / MIMIC (clinical time series; CREDENTIALED -- register, do not scrape):"
Write-Host "    https://physionet.org"
Write-Host "  FDA FAERS (adverse-event reports, public):"
Write-Host "    https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers"
Write-Host "  OSP PBPK Model Library (whole-body PBPK models):"
Write-Host "    https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library"
Write-Host "  DDMoRe model repository (curated pharmacometric models):"
Write-Host "    https://ddmore.eu/models-tools"
Write-Host ""
Write-Host "Bigger SYNTHETIC population (no download):"
Write-Host "  python scripts/make_synthetic.py --patients 100000"
