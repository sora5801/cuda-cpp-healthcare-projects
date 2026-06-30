# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.24 : SAXS / SANS Data-Driven Structure Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Real SAXS curves and PDB models are
# downloaded by the user from the public banks below; this script only prints
# guidance and defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.24 -- SAXS / SANS Data-Driven Structure Modeling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The committed tiny SYNTHETIC sample (data/sample/saxs_sample.txt) already"
Write-Host "lets the demo run offline. No real data is required to study this project."
Write-Host ""
Write-Host "To work with REAL small-angle scattering data, fetch it yourself from:"
Write-Host "  * SASBDB  -- curated SAXS/SANS curves + models : https://www.sasbdb.org"
Write-Host "             (download a .dat file: columns are 'q  I  sigma', the same"
Write-Host "              layout as our sample's curve section)"
Write-Host "  * RCSB PDB -- atomic models to forward-model     : https://www.rcsb.org"
Write-Host "  * BIOISIS  -- SAXS benchmark database (verify the current URL)"
Write-Host ""
Write-Host "Converting a real .pdb to our text format means: read ATOM records, map"
Write-Host "each element to an electron count (or a proper q-dependent form factor; see"
Write-Host "THEORY.md), then append a .dat curve from SASBDB as the 'q I_exp sigma' block."
Write-Host ""
Write-Host "For a larger SYNTHETIC problem instead, run:"
Write-Host "    python scripts/make_synthetic.py --atoms 500 --nq 60 --out data/sample/big.txt"
Write-Host ""
Write-Host "Idempotent real-fetch pattern (when you wire one up):"
Write-Host "    1) skip the download if the file already exists with the right checksum"
Write-Host "    2) print source URL + expected size + SHA256"
Write-Host "    3) for credentialed sets, print registration instructions ONLY"
