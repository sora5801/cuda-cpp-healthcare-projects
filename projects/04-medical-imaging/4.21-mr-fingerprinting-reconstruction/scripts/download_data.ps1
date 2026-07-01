# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.21 -- MR Fingerprinting Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# how to obtain each dataset, and NEVER bypasses credentials/registration. The
# real MRF datasets below require registration/agreements, so this script only
# prints instructions + links and defers to scripts/make_synthetic.py for an
# offline stand-in. The committed tiny sample already runs the demo with zero
# downloads.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.21 -- MR Fingerprinting Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a TINY committed SYNTHETIC sample (data/sample/mrf_sample.txt)"
Write-Host "that runs the demo offline. Real MR Fingerprinting datasets require"
Write-Host "registration and are NOT redistributed here. To obtain them yourself:"
Write-Host ""
Write-Host "  1) fastMRI (includes qMRI/MRF-style data) -- https://fastmri.org/"
Write-Host "     Register and accept the data use agreement; download instructions"
Write-Host "     are provided after approval. (Verify the exact MRF subset URL there.)"
Write-Host ""
Write-Host "  2) Cleveland Clinic MRF dataset -- search IEEE DataPort (https://ieee-dataport.org/)"
Write-Host "     for 'MR Fingerprinting'; access terms vary per collection (verify URL)."
Write-Host ""
Write-Host "  3) qMRI.org quantitative-MRI resources -- https://qmri.org/ (verify URL)."
Write-Host ""
Write-Host "  4) Synthetic phantoms -- generate BrainWeb/XCAT-style ground truth locally."
Write-Host ""
Write-Host "For a LARGER synthetic problem (more voxels / a bigger dictionary), run:"
Write-Host "    python scripts/make_synthetic.py --V 4096"
Write-Host ""
Write-Host "When wiring a real dataset, follow this idempotent pattern:"
Write-Host "    1) skip download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
Write-Host ""
Write-Host "[download_data] No files downloaded (by design). The demo runs on the"
Write-Host "[download_data] committed synthetic sample."
