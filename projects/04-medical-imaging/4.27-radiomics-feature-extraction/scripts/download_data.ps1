# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.27 : Radiomics Feature Extraction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL, and
# NEVER bypasses credentials/registration. Real radiomics data is large DICOM
# imaging with segmentation masks (TCIA, GDC); those must not be redistributed
# here, so this script only prints instructions + links and defers to
# scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.27 -- Radiomics Feature Extraction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed tiny SYNTHETIC sample (data/sample/radiomics_sample.txt) is"
Write-Host "enough to build and run the demo offline. No download is required."
Write-Host ""
Write-Host "Real radiomics cohorts (imaging + segmentations) live at:"
Write-Host "  * TCIA NSCLC-Radiomics (422 lung CTs + survival):"
Write-Host "      https://www.cancerimagingarchive.net/collection/nsclc-radiomics/"
Write-Host "  * QIN-HEADNECK (head & neck RT), RIDER Breast MRI -- via TCIA."
Write-Host "  * TCGA imaging + clinical: https://portal.gdc.cancer.gov/"
Write-Host ""
Write-Host "How to use real data with this project:"
Write-Host "  1) Download a collection from TCIA (respect each collection's LICENSE;"
Write-Host "     some require a Data Use Agreement -- do NOT bypass it)."
Write-Host "  2) Read the CT/PET/MRI volume and its ROI segmentation with pydicom /"
Write-Host "     SimpleITK, crop to the ROI bounding box, quantize intensities to Ng"
Write-Host "     levels, and write the 'nx ny nz Ng' + intensities + mask text format"
Write-Host "     that data/README.md documents (this conversion is an exercise)."
Write-Host ""
Write-Host "For a larger SYNTHETIC problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --nx 64 --ny 64 --nz 48 --ng 16"
