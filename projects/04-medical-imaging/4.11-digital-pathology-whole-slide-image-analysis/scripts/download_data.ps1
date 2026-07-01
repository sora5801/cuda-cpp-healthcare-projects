# ===========================================================================
# scripts/download_data.ps1  --  Real WSI-data pointers (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 4.11 : Digital Pathology / Whole-Slide Image Analysis
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. Whole-slide images are multi-gigabyte
# and their public repositories require a (free) account and agreeing to a data-
# use agreement, so this script does NOT auto-download; it prints where to get
# the data and how to turn it into this project's tile-feature-bag format. The
# committed synthetic sample already lets the demo run offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.11 -- Digital Pathology / Whole-Slide Image Analysis"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project consumes a BAG of tile FEATURE vectors per slide, not raw"
Write-Host "pixels. The real pipeline is: download WSIs -> tile + tissue-detect ->"
Write-Host "run a frozen CNN/ViT encoder per tile -> save the N x D features."
Write-Host ""
Write-Host "Public WSI datasets (free account + data-use agreement required):"
Write-Host "  TCGA slides (GDC) : https://portal.gdc.cancer.gov/"
Write-Host "  CAMELYON16/17     : https://camelyon17.grand-challenge.org/"
Write-Host "  TUPAC16           : http://tupac.tue-image.nl/"
Write-Host ""
Write-Host "Tools to read WSIs and extract features:"
Write-Host "  OpenSlide         : https://openslide.org/           (read .svs/.tif pyramids)"
Write-Host "  CLAM              : https://github.com/mahmoodlab/CLAM (tiling + feature bags + MIL)"
Write-Host "  UNI encoder       : https://github.com/mahmoodlab/UNI  (pretrained ViT features)"
Write-Host ""
Write-Host "Export each slide as 'N D label' then N rows of D features (D must equal"
Write-Host "FEAT_DIM in src/wsi.h). See data/README.md for the exact format."
Write-Host ""
Write-Host "No download needed to run the demo. Bigger SYNTHETIC bag:"
Write-Host "  python scripts/make_synthetic.py --n 20000"
