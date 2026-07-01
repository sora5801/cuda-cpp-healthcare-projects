# ===========================================================================
# scripts/download_data.ps1  --  Real multi-site imaging pointers (Windows)
# ---------------------------------------------------------------------------
# Project 4.25 : Image Harmonization Across Scanners/Sites.
#
# There is NOTHING to auto-download: every real multi-site imaging dataset below
# requires registration / a data-use agreement, and most FORBID redistribution.
# This script NEVER attempts to bypass credentials (CLAUDE.md §8). It prints the
# official links and how to turn extracted features into our loader format; the
# committed synthetic sample lets the demo run offline in the meantime.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 4.25 -- Image Harmonization Across Scanners/Sites"
Write-Host ""
Write-Host "ComBat operates on EXTRACTED FEATURES (e.g. FreeSurfer regional volumes /"
Write-Host "cortical thickness, or radiomic features), not raw voxels. Export a feature"
Write-Host "table into the format in data/README.md:"
Write-Host "    line 1 : N P B C"
Write-Host "    line 2 : batch (scanner/site) label per sample"
Write-Host "    N lines: C covariate values per sample (age, sex, ...)"
Write-Host "    P lines: N feature values per line"
Write-Host ""
Write-Host "Public multi-site imaging datasets (registration / DUA required):"
Write-Host "  ABIDE (autism, multi-site) : http://fcon_1000.projects.nitrc.org/indi/abide/"
Write-Host "  ADNI  (Alzheimer's)        : https://adni.loni.usc.edu/"
Write-Host "  IXI   (multi-site brain MRI): https://brain-development.org/ixi-dataset/"
Write-Host "  UK Biobank imaging         : https://www.ukbiobank.ac.uk/"
Write-Host ""
Write-Host "Reference implementation to compare against:"
Write-Host "  NeuroComBat : https://github.com/Jfortin1/ComBatHarmonization"
Write-Host ""
Write-Host "No download needed -- generate a bigger SYNTHETIC set instead:"
Write-Host "  python scripts/make_synthetic.py --p 200 --b 4 --n 120"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
