# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.28 : GPU-Accelerated DRR Generation for 2D/3D Registration
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL and
# NEVER bypasses credentials/registration. The DRR demo runs fully on the
# committed SYNTHETIC phantom (data/sample/ct_volume_sample.txt), so there is no
# mandatory download. This script points at real CT sources and defers to
# make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.28 -- GPU-Accelerated DRR Generation for 2D/3D Registration"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed sample (data/sample/ct_volume_sample.txt) is SYNTHETIC and is all"
Write-Host "the demo needs. No download is required to build, run, or verify this project."
Write-Host ""
Write-Host "To experiment on REAL CT volumes, convert a DICOM series into this loader's"
Write-Host "text format (header 'nx ny nz sx sy sz' then nx*ny*nz Hounsfield Units,"
Write-Host "row-major [z][y][x]) using e.g. pydicom or SimpleITK. Public sources:"
Write-Host "  * TCIA (The Cancer Imaging Archive): https://www.cancerimagingarchive.net/"
Write-Host "      prostate/lung CT collections (mostly CC-BY)."
Write-Host "  * Gold Atlas male-pelvis MR/CT (verify URL): https://www.goldenatlasproject.com/"
Write-Host "  * AAPM TG-132 image-registration test cases."
Write-Host "  * Clinical CBCT + kV portal images: institutional IRB only -- NOT redistributed."
Write-Host ""
Write-Host "For a larger SYNTHETIC volume (no download, no credentials), run:"
Write-Host "    python scripts/make_synthetic.py --n 128      # 128^3 phantom"
Write-Host ""
Write-Host "This script intentionally downloads nothing automatically: the public sets are"
Write-Host "large and/or require accepting a data-use agreement, which must be done by a"
Write-Host "human (CLAUDE.md section 8). It will never bypass that step."
