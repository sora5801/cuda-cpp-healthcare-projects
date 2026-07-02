# ===========================================================================
# scripts/download_data.ps1  --  "Fetch the FULL dataset" (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.9 : Agent-Based Tissue / Immune Simulation
#
# This project is a SIMULATION: its input is a scenario parameter file, not a
# measured dataset, so there is nothing to download to run the demo. This script
# (per CLAUDE.md §8) explains where REAL tissue/immune data live for those who
# want to calibrate the model, and NEVER bypasses any registration.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.9 -- Agent-Based Tissue / Immune Simulation"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project generates its own tissue state; NO download is needed to run"
Write-Host "the demo. The committed sample (data/sample/tissue_params.txt) is enough."
Write-Host ""
Write-Host "For a larger SYNTHETIC scenario, regenerate the parameter file:"
Write-Host "    python scripts/make_synthetic.py --gx 64 --gy 64 --n-tumor 400 --n-immune 300 --steps 800"
Write-Host ""
Write-Host "To CALIBRATE cell states / immune landscapes against real data, see"
Write-Host "(each requires its own registration / license -- follow their terms):"
Write-Host "  * CancerSEA single-cell functional states : http://biocc.hrbmu.edu.cn/CancerSEA/"
Write-Host "  * TCGA pan-cancer immune landscape        : https://portal.gdc.cancer.gov"
Write-Host "  * MIBI/IMC imaging mass cytometry         : various Zenodo deposits"
Write-Host "  * TCIA immunotherapy imaging              : https://www.cancerimagingarchive.net"
Write-Host ""
Write-Host "These are NOT auto-downloaded: they are large, credentialed, and their"
Write-Host "licenses forbid blind redistribution. Educational use only -- not clinical."
