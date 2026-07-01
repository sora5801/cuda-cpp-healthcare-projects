# ===========================================================================
# scripts/download_data.ps1  --  Fetch / point to the FULL dataset (Windows)
# ---------------------------------------------------------------------------
# Project 5.5 : Deformable Dose Accumulation & Adaptive Radiotherapy
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs +
# licensing notes, and NEVER bypasses credentials/registration. The demo needs
# nothing downloaded -- scripts/make_synthetic.py writes a tiny offline sample.
# This script tells a learner who wants REAL ART data where to get it and how to
# shape it for this project.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.5 -- Deformable Dose Accumulation & Adaptive Radiotherapy"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC offline sample (data/sample/art_case.txt);"
Write-Host "the demo needs nothing downloaded. The datasets below are real-world DIR /"
Write-Host "ART benchmarks -- each requires you to accept a data-use license yourself,"
Write-Host "so this script only PRINTS pointers. It never bypasses any registration."
Write-Host ""
Write-Host "  * DIR-Lab 4D-CT lung  : https://www.dir-lab.com/"
Write-Host "      Respiratory 4D-CT phase pairs with expert landmarks (the gold-standard"
Write-Host "      target-registration-error benchmark for DIR)."
Write-Host "  * AAPM TG-132 DIR      : https://www.aapm.org/pubs/reports/RPT_132.pdf"
Write-Host "      The clinical QA reference for DIR + deformable dose accumulation."
Write-Host "  * TCIA CT-on-rails/CBCT: https://www.cancerimagingarchive.net/"
Write-Host "      Planning CT + daily CBCT collections for adaptive-radiotherapy studies."
Write-Host "  * CREATIS lung phantom : https://www.creatis.insa-lyon.fr/"
Write-Host "      A deformable lung phantom with a known ground-truth motion field."
Write-Host ""
Write-Host "To use real data with THIS teaching project, export one 2-D slice pair:"
Write-Host "  planning image, daily image, planning dose, daily dose  (all same nx x ny),"
Write-Host "normalize images to [0,1] and doses to Gy, and write them in the sample's"
Write-Host "text format (see data/README.md). DO NOT commit patient-derived data."
Write-Host ""
Write-Host "For a larger SYNTHETIC problem instead:"
Write-Host "  python scripts/make_synthetic.py --nx 256 --ny 256 --shift 12.0"
