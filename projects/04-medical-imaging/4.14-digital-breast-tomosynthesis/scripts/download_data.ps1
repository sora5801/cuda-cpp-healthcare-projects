# ===========================================================================
# scripts/download_data.ps1  --  Fetch / point to the FULL dataset (Windows)
# ---------------------------------------------------------------------------
# Project 4.14 -- Digital Breast Tomosynthesis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real DBT/mammography datasets are
# credentialed or non-redistributable, so this script prints how to obtain them
# and defers to scripts/make_synthetic.py for an offline synthetic stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 4.14 -- Digital Breast Tomosynthesis"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed tiny sample (data/sample/dbt_sample.txt) is SYNTHETIC and is"
Write-Host "all the demo needs -- no download required. Real DBT/mammography data:"
Write-Host ""
Write-Host "  * CBIS-DDSM (curated mammograms via TCIA, open):"
Write-Host "      https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM"
Write-Host "  * BCS-DBT (Duke tomosynthesis challenge, true DBT projections):"
Write-Host "      https://bcs-dbt.grand-challenge.org/"
Write-Host "  * VinDr-Mammo (PhysioNet, CREDENTIALED -- requires a signed DUA):"
Write-Host "      https://physionet.org/content/vindr-mammo/1.0.0/"
Write-Host "  * OPTIMAM / OMI-DB (access via ICR UK, CREDENTIALED)."
Write-Host ""
Write-Host "This script does NOT bypass any registration/credential wall. For the"
Write-Host "credentialed sets, register at the link, accept the licence, and place the"
Write-Host "files under data/ yourself. For a larger SYNTHETIC problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --img 128 --angles 21 --det 160"
