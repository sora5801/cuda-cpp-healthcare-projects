# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.22 : Bone Remodeling Simulation   (REDUCED-SCOPE teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This teaching model runs entirely on
# the committed SYNTHETIC parameter file (data/sample/bone_params.txt) plus
# scripts/make_synthetic.py, so there is nothing to download to run the demo.
# The pointers below are for learners who want to drive a production voxel-FEM
# remodeling pipeline on real bone microCT geometry.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.22 -- Bone Remodeling Simulation"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project needs NO download: the committed synthetic sample at"
Write-Host "  data/sample/bone_params.txt"
Write-Host "is sufficient to build and run the demo offline. For a larger SYNTHETIC"
Write-Host "problem, run:"
Write-Host "    python scripts/make_synthetic.py --nx 64 --ny 48 --load-x0 28 --load-x1 35"
Write-Host ""
Write-Host "Real bone-imaging datasets you could adapt a voxel-FEM pipeline to"
Write-Host "(segment a microCT stack into a bone/marrow voxel mask, then remodel):"
Write-Host "  * OsteoArthritis Initiative (OAI): https://nda.nih.gov/oai/  (registration required)"
Write-Host "  * PhysioNet bone datasets:         https://physionet.org     (credentialed use for some)"
Write-Host "  * BoneJ morphometric examples:     https://bonej.org"
Write-Host "  * MICCAI bone segmentation:        https://grand-challenge.org"
Write-Host ""
Write-Host "Respect every license; NEVER bypass registration. If redistribution is"
Write-Host "forbidden, keep using the synthetic sample (see data/README.md)."
