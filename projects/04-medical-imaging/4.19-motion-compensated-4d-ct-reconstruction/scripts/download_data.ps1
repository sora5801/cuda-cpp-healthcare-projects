# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL 4D-CT data (Windows)
# ---------------------------------------------------------------------------
# Project 4.19 : Motion-Compensated 4D-CT Reconstruction
#
# CONTRACT (CLAUDE.md §8): documented, and NEVER bypasses credentials/registration.
# This project's committed sample is synthetic, so this script downloads nothing;
# it prints where the real datasets live and how production tools consume them.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.19 -- Motion-Compensated 4D-CT Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Options for real 4D-CT data (a phase-binned sinogram / projection set):"
Write-Host "  * DIR-Lab 4D-CT lung (https://www.dir-lab.com/) -- 10 cases with expert"
Write-Host "    landmark pairs; the standard deformable-registration benchmark."
Write-Host "  * TCIA 4D-CT lung radiotherapy (https://www.cancerimagingarchive.net) --"
Write-Host "    real DICOM; registration may be required. This script does NOT bypass it."
Write-Host "  * POPI model (https://www.creatis.insa-lyon.fr/rio/popi-model) -- a"
Write-Host "    point-validated breathing 4D-CT dataset."
Write-Host "  * Toolkits with 4D reconstruction + sample data: RTK (ROOSTER), ASTRA,"
Write-Host "    TIGRE, Plastimatch. Use them to forward-project into this sinogram format."
Write-Host ""
Write-Host "Offline stand-in (no download, reproducible, SYNTHETIC):"
Write-Host "  python scripts/make_synthetic.py --phases 10 --ang-phase 12 --det 257 --img 128"
Write-Host ""
Write-Host "When wiring a real dataset, keep it idempotent:"
Write-Host "  1) skip download if the file already exists with the right checksum"
Write-Host "  2) print source URL + expected size + SHA256"
Write-Host "  3) for credentialed sets, print registration instructions ONLY"
