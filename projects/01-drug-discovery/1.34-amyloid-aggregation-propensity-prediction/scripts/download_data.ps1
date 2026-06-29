# ===========================================================================
# scripts/download_data.ps1  --  Point at the real datasets (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.34 : Amyloid / Aggregation Propensity Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The demo runs entirely on the tiny
# SYNTHETIC FASTA committed in data/sample/, so no download is required to learn
# from this project. This script tells you where the real curated aggregation
# datasets live and how to use them with the binary.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.34 -- Amyloid / Aggregation Propensity Prediction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed tiny SYNTHETIC sample (data/sample/amyloid_sample.fasta)"
Write-Host "is enough to run the demo offline. The binary accepts any FASTA file:"
Write-Host "    amyloid-aggregation-propensity-prediction.exe <your.fasta>"
Write-Host ""
Write-Host "Real curated aggregation datasets (study these; respect each license):"
Write-Host "  * AmyPro    -- curated amyloidogenic protein regions (FASTA + annotations)"
Write-Host "                 https://amypro.net   (downloadable; cite Varadi et al. 2018)"
Write-Host "  * WALTZ-DB 2.0 -- experimental hexapeptide amyloid/non-amyloid labels"
Write-Host "                 https://waltzdb.switchlab.org"
Write-Host "  * EMDB fibril cryo-EM maps (structural validation of predicted APRs)"
Write-Host "                 https://www.ebi.ac.uk/emdb/"
Write-Host ""
Write-Host "How to use a real set with this teaching tool:"
Write-Host "  1) Download the sequences as a plain FASTA file (one '>' header per protein)."
Write-Host "  2) Run the binary on it; it scans every sequence and ranks them by APR."
Write-Host "  3) Compare the predicted hot spots against the database's annotated"
Write-Host "     amyloidogenic regions -- that is the natural next exercise (README)."
Write-Host ""
Write-Host "Idempotent download pattern to follow if you script a real fetch:"
Write-Host "  - skip if the file already exists with the expected SHA256;"
Write-Host "  - print source URL + expected size + checksum before downloading;"
Write-Host "  - for any credentialed source, print registration instructions ONLY."
Write-Host ""
Write-Host "For a larger SYNTHETIC batch to stress the GPU path, regenerate the sample"
Write-Host "or duplicate sequences in your own FASTA (the kernel batches them all)."
