# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.16 : ADMET / Toxicity Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The public ADMET sets below are
# downloadable directly, but turning molecules into the descriptor format this
# project expects requires a featurizer (RDKit / Chemprop) you run yourself, so
# this script prints the recipe and links rather than fabricating data. The
# committed synthetic sample in data/sample/ is enough to run the demo offline.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.16 -- ADMET / Toxicity Prediction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Public ADMET / toxicity datasets (fetch yourself; respect each license):"
Write-Host "  * Tox21 (12 endpoints, ~8k compounds): https://tripod.nih.gov/tox21/"
Write-Host "  * TDC ADMET benchmark group:           https://tdcommons.ai/benchmark/admet_group/overview/"
Write-Host "  * ClinTox (FDA approved vs failed):    https://moleculenet.org"
Write-Host "  * DILI (drug-induced liver injury):    search current literature for a redistributable release"
Write-Host ""
Write-Host "To turn SMILES/molecules into the <name> <descriptor...> format this project reads:"
Write-Host "  1) featurize with RDKit descriptors or Chemprop D-MPNN features"
Write-Host "  2) write one line per molecule: '<name> <x_0> ... <x_{D-1}>' (D = ADMET_D in src/admet_core.h)"
Write-Host "  3) prepend the M trained endpoint models as '<endpoint> <bias> <w_0> ... <w_{D-1}>'"
Write-Host ""
Write-Host "No credentialed download is attempted. For an OFFLINE synthetic stand-in:"
Write-Host "    python scripts/make_synthetic.py --n 1000000"
Write-Host ""
Write-Host "Idempotency pattern when wiring a real fetch: skip if the file already exists"
Write-Host "with the right SHA256; print source URL + expected size + checksum; for"
Write-Host "credentialed sets print registration instructions ONLY (never bypass them)."
