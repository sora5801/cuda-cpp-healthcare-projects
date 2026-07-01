# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.2 -- Whole-Heart Digital Twin   (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. This project's
# demo needs NO external data -- its input is a tiny synthetic ensemble config
# (scripts/make_synthetic.py). The datasets below are the REAL-WORLD sources a
# full patient-specific twin is built from; most require registration, so we
# only print links and instructions.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.2 -- Whole-Heart Digital Twin"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project needs NO download: the demo runs on the tiny"
Write-Host "synthetic ensemble config in data/sample/heart_ensemble.txt."
Write-Host "Regenerate or resize it with:"
Write-Host "    python scripts/make_synthetic.py --n 256"
Write-Host ""
Write-Host "REAL-WORLD datasets a full cardiac digital twin is built from"
Write-Host "(geometry + fibers + calibration targets); most need registration:"
Write-Host "  * UK Biobank Cardiac MRI (100k+ cine CMR) -- https://www.ukbiobank.ac.uk  [application required]"
Write-Host "  * Zenodo Synthetic Biventricular Heart Meshes (1000 meshes) -- https://zenodo.org/records/4506930  [open, CC-BY]"
Write-Host "  * Visible Human Project (CT/MRI/cryosection) -- https://www.nlm.nih.gov/research/visible/visible_human.html  [license/registration]"
Write-Host "  * ACDC MICCAI (100-patient CMR segmentations) -- https://www.creatis.insa-lyon.fr/Challenge/acdc/  [registration]"
Write-Host ""
Write-Host "None are fetched automatically: credentialed sets must be obtained by"
Write-Host "the user under their own agreement (CLAUDE.md section 8)."
