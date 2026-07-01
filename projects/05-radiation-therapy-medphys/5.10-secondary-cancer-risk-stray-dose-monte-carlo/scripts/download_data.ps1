# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 5.10 : Secondary Cancer Risk & Stray-Dose Monte Carlo
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. The
# real inputs (ICRP-110 voxel phantoms, NIST XCOM cross-sections, TCIA CTs) are
# large and/or registration-gated, so this script prints instructions + links and
# defers to scripts/make_synthetic.py for the offline stand-in the demo uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 5.10 -- Secondary Cancer Risk & Stray-Dose Monte Carlo"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC committed sample (data/sample/phantom.txt),"
Write-Host "so no download is required to run the demo."
Write-Host ""
Write-Host "Real datasets (registration/attribution required -- fetch by hand):"
Write-Host "  * ICRP 110 voxel phantoms (adult male/female):"
Write-Host "      https://www.icrp.org/publication.asp?id=ICRP%20Publication%20110"
Write-Host "  * NIST XCOM photon cross-sections:"
Write-Host "      https://www.nist.gov/pml/xcom-photon-cross-sections"
Write-Host "  * TCIA proton-therapy planning CTs (account + attribution):"
Write-Host "      https://www.cancerimagingarchive.net/"
Write-Host ""
Write-Host "For a larger SYNTHETIC problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --histories 2000000 --seed 7"
