# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.14 : Multi-Scale Physiological Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This project is SIMULATION-ONLY: the demo
# needs no download (the tiny synthetic sample in data/sample/ is enough). This
# script therefore just points you at the real multi-scale model repositories a
# production VPH workflow would draw cell/tissue models from.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.14 -- Multi-Scale Physiological Modeling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project is simulation-only; no dataset download is required to run"
Write-Host "the demo. The committed synthetic sample (data/sample/cable.txt) suffices."
Write-Host ""
Write-Host "Real multi-scale physiology model repositories (study these):"
Write-Host "  * Physiome Model Repository (CellML cell models):"
Write-Host "      https://models.physiomeproject.org"
Write-Host "  * BioModels Database (systems-biology / ODE models):"
Write-Host "      https://www.ebi.ac.uk/biomodels"
Write-Host "  * OpenCMISS examples (multi-scale FEM setups):"
Write-Host "      https://github.com/OpenCMISS/examples"
Write-Host "  * UK Biobank multi-modal phenotyping (CREDENTIALED -- do NOT bypass;"
Write-Host "    apply for access at):"
Write-Host "      https://www.ukbiobank.ac.uk"
Write-Host ""
Write-Host "For a larger SYNTHETIC cable (more nodes / longer run), use:"
Write-Host "  python scripts/make_synthetic.py --n 512 --steps 20000"
