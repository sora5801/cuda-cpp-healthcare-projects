# ===========================================================================
# scripts/download_data.ps1  --  Fetch / locate the FULL dataset (PowerShell)
# ---------------------------------------------------------------------------
# Project 4.22 : Quantitative Susceptibility Mapping (QSM)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs +
# licensing, and NEVER bypasses credentials/registration. Every real QSM dataset
# below requires registration or carries redistribution limits, so this script
# only PRINTS instructions and links; the committed synthetic sample
# (data/sample/field_map.txt) lets the demo run offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.22 -- Quantitative Susceptibility Mapping (QSM)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The demo runs on the committed SYNTHETIC sample (data/sample/field_map.txt)."
Write-Host "No download is required to build, run, or study this project."
Write-Host ""
Write-Host "To study REAL QSM data, use one of these sources (each has its own"
Write-Host "license and most require registration -- respect them; we do not"
Write-Host "redistribute any of them):"
Write-Host ""
Write-Host "  * QSM Reconstruction Challenge 2.0 (benchmark data + reference recons):"
Write-Host "      https://doi.org/10.1101/2020.11.25.397695   (data on Zenodo)"
Write-Host "  * HCP 7T multi-echo GRE (Human Connectome Project):"
Write-Host "      https://db.humanconnectome.org/              (registration required)"
Write-Host "  * AHEAD ultra-high-field 7T lifespan database (Amsterdam)."
Write-Host "  * UK Biobank (credentialed):"
Write-Host "      https://www.ukbiobank.ac.uk/                 (application required)"
Write-Host ""
Write-Host "After obtaining a LOCAL FIELD MAP (phase unwrapped + background removed),"
Write-Host "export the 3-D volume to this project's text format:"
Write-Host "    line 1: 'nx ny nz'"
Write-Host "    then nx*ny*nz field-shift values, x fastest then y then z"
Write-Host "and pass its path to the executable. See data/README.md for details."
Write-Host ""
Write-Host "For a larger SYNTHETIC field map instead, run:"
Write-Host "    python scripts/make_synthetic.py --nx 24 --ny 24 --nz 16"
