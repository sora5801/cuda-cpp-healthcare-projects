# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.20 -- Heterogeneous Cryo-EM Reconstruction (3D Variability)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. This project is a
# REDUCED-SCOPE teaching version that runs on a committed SYNTHETIC sample, so
# this script only prints pointers to the real datasets and defers to
# scripts/make_synthetic.py for a larger offline stand-in. It downloads nothing.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.20 -- Heterogeneous Cryo-EM Reconstruction (3D Variability)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project ships a SYNTHETIC sample (data/sample/volumes.txt);"
Write-Host "no download is required to run the demo. The real heterogeneous cryo-EM"
Write-Host "datasets the catalog points at are large and need preprocessing:"
Write-Host ""
Write-Host "  EMPIAR-10180 (spliceosome), EMPIAR-10076 (80S ribosome),"
Write-Host "  EMPIAR-10028 (TRPV1)              -> https://www.ebi.ac.uk/empiar/"
Write-Host "  cryoDRGN benchmark sets + tooling -> https://github.com/ml-struct-bio/cryodrgn"
Write-Host ""
Write-Host "  EMPIAR entries are openly downloadable but tens-to-hundreds of GB; turning"
Write-Host "  a particle stack into per-particle volumes (CTF, poses, back-projection)"
Write-Host "  is upstream of this project. Respect each dataset's license."
Write-Host ""
Write-Host "  For a larger SYNTHETIC problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --n 64 --g 8"
