# ===========================================================================
# scripts/download_data.ps1  --  Real OCT dataset pointers (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.12 : Optical Coherence Tomography Processing (SD-OCT reconstruction)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL, and
# NEVER bypasses credentials/registration. The public OCT datasets below ship
# PROCESSED B-scan IMAGES (already reconstructed), not the vendor RAW spectra our
# reconstruction consumes -- raw interferograms are device-specific and rarely
# public. So this script downloads nothing; it points at the datasets for the
# downstream tasks (segmentation, classification) and defers to
# scripts/make_synthetic.py for a runnable RAW-spectrum stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.12 -- Optical Coherence Tomography Processing"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Public OCT datasets (PROCESSED B-scans / volumes, for segmentation/classification):"
Write-Host "  OCTDL     : https://www.nature.com/articles/s41597-024-03182-7  (2,064 labeled B-scans)"
Write-Host "  Duke DME  : https://people.duke.edu/~sf59/Chiu_BOE_2012_dataset.htm  (110 annotated volumes)"
Write-Host "  OCTA-500  : https://arxiv.org/abs/2012.07261  (OCT angiography volumes with labels)"
Write-Host ""
Write-Host "NOTE: those provide reconstructed images, not vendor RAW spectra. This"
Write-Host "project reconstructs FROM raw spectra, so the committed sample is synthetic"
Write-Host "raw interferograms (scripts/make_synthetic.py). Raw-spectrum access requires"
Write-Host "the OCT device SDK (Thorlabs/Bioptigen/Heidelberg) -- follow the vendor's"
Write-Host "terms; this script will not bypass any registration."
Write-Host ""
Write-Host "Bigger SYNTHETIC B-scan (no download):"
Write-Host "  python scripts/make_synthetic.py --n-ascan 128 --n-spec 1024"
