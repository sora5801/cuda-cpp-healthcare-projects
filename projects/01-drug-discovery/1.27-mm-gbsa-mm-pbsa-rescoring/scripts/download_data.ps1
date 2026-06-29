# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.27 : MM-GBSA / MM-PBSA Rescoring
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs +
# notes, and NEVER bypasses credentials/registration. The committed tiny sample
# in data/sample/ already lets the demo run offline; this script only points at
# the real, credentialed datasets and defers to make_synthetic.py for a larger
# offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.27 -- MM-GBSA / MM-PBSA Rescoring"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project's REAL inputs are MD trajectories of protein-ligand complexes"
Write-Host "plus a force field (charges, LJ params, Born radii). Those sources require"
Write-Host "registration and carry licenses, so this script does NOT auto-download them."
Write-Host ""
Write-Host "  * PDBbind (complexes + measured affinities) : http://www.pdbbind.org.cn"
Write-Host "  * CASF-2016 (scoring benchmark / core set)   : http://www.pdbbind.org.cn/casf.php"
Write-Host "  * ChEMBL (bioactivity data)                  : https://www.ebi.ac.uk/chembl/"
Write-Host "  * AMBER MM-GBSA tutorials (ready trajectories): https://ambermd.org/tutorials/"
Write-Host ""
Write-Host "  Respect each dataset's license; for credentialed sets, register at the URL"
Write-Host "  above -- this script will not bypass any login."
Write-Host ""
Write-Host "  The committed sample (data/sample/complex_sample.txt) runs the demo offline."
Write-Host "  For a LARGER synthetic problem (e.g. 64 snapshots), run:"
Write-Host "    python scripts/make_synthetic.py --snapshots 64"
