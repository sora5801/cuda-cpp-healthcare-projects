# ===========================================================================
# scripts/download_data.ps1  --  Realistic virtual-population pointers (Windows)
# ---------------------------------------------------------------------------
# Project 6.26 : Virtual Population Generation & Sensitivity Analysis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. There is nothing to download for the demo -- the
# study is generated from data/sample/vpop_config.txt. This script only prints
# where the REAL physiology/PBPK data lives and defers to make_synthetic.py for
# a larger offline study.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.26 -- Virtual Population Generation & Sensitivity Analysis"
Write-Host ""
Write-Host "There is no file to download: the demo's virtual population is"
Write-Host "generated deterministically from data/sample/vpop_config.txt."
Write-Host ""
Write-Host "For a REALISTIC virtual population + sensitivity workflow, use:"
Write-Host "  NHANES physiology  : https://www.cdc.gov/nchs/nhanes/"
Write-Host "  WHO growth data    : https://www.who.int/tools/growth-reference-data-for-5to19-years"
Write-Host "  OSP PBPK library   : https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library"
Write-Host "  FDA drug-label PK  : https://www.fda.gov/drugs"
Write-Host "  SALib (reference)  : https://github.com/SALib/SALib"
Write-Host ""
Write-Host "These are externally licensed; respect each source's terms. This"
Write-Host "script does NOT attempt to bypass any registration."
Write-Host ""
Write-Host "Bigger SYNTHETIC study (no download):"
Write-Host "  python scripts/make_synthetic.py --N 16384"
Write-Host ""
Write-Host "Target data dir: $DataDir"
