# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.9 : Image Denoising & Restoration  (Non-Local Means)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. Every real dataset
# for this project is either license-restricted or credentialed, so this script
# prints instructions + links ONLY and defers to scripts/make_synthetic.py for an
# offline stand-in. The committed data/sample/ is enough to run the demo.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.9 -- Image Denoising & Restoration (Non-Local Means)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed tiny SYNTHETIC sample (data/sample/phantom_sample.txt) is"
Write-Host "enough to build and run the demo offline. The REAL medical datasets below"
Write-Host "are license-restricted or credentialed -- this script only prints how to"
Write-Host "obtain them; it never bypasses any registration (CLAUDE.md section 8)."
Write-Host ""
Write-Host "  1) 2016 AAPM Low-Dose CT Grand Challenge (quarter/full-dose CT pairs)"
Write-Host "       https://www.aapm.org/grandchallenge/lowdosect/"
Write-Host "       -> agree to the challenge data-use terms, then download the DICOM pairs."
Write-Host "  2) NLST (National Lung Screening Trial) chest CT via TCIA"
Write-Host "       https://www.cancerimagingarchive.net/"
Write-Host "       -> requires a TCIA account + data-use agreement."
Write-Host "  3) Fluorescence Microscopy Noise Dataset (for Noise2Void)"
Write-Host "       https://github.com/juglab/n2v"
Write-Host "  4) SIDD smartphone image-noise dataset (non-medical sanity check)"
Write-Host ""
Write-Host "For a larger SYNTHETIC problem you can generate right now:"
Write-Host "    python scripts/make_synthetic.py --size 128 --sigma 0.10"
Write-Host ""
Write-Host "When wiring a real dataset later, keep this idempotent pattern:"
Write-Host "    1) skip the download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
