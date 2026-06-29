# ===========================================================================
# scripts/download_data.ps1  --  How to get REAL CT data (Windows)
# ---------------------------------------------------------------------------
# Project 4.01 : CT Reconstruction (Filtered Backprojection)
# Prints how to obtain a sinogram/phantom; downloads nothing. See section 8.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 4.01 -- CT Reconstruction (Filtered Backprojection)"
Write-Host ""
Write-Host "Options for real/standard data:"
Write-Host "  * Shepp-Logan digital phantom -- generate with ASTRA/TIGRE, forward-project,"
Write-Host "    then write the sinogram in data/README.md format."
Write-Host "  * TCIA (https://www.cancerimagingarchive.net) -- real DICOM CT projection data"
Write-Host "    (registration may be required; this script does NOT bypass it)."
Write-Host "  * Reconstruction toolkits with sample data: RTK, ASTRA, TIGRE, Plastimatch."
Write-Host ""
Write-Host "Offline stand-in (no download, reproducible):"
Write-Host "  python scripts/make_synthetic.py --angles 360 --det 367 --img 256"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
