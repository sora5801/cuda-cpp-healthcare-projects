# ===========================================================================
# scripts/download_data.ps1  --  Pointers to the real datasets (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.14 : Conformer Ensemble Generation
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# access notes, and NEVER bypasses credentials/registration.
#
# IMPORTANT: this teaching demo is SELF-CONTAINED -- the molecule is fixed in
# src/conformer.h and the committed data/sample/conformer_params.txt is all the
# demo needs. There is therefore NOTHING to download to run this project. The
# datasets below are what you would use to VALIDATE a production conformer
# generator (compare generated shapes/energies to crystallographic / DFT data).
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.14 -- Conformer Ensemble Generation"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project needs NO download to run: data/sample/conformer_params.txt"
Write-Host "plus the compile-time molecule in src/conformer.h are sufficient."
Write-Host ""
Write-Host "Real-world reference datasets (for validating a production generator):"
Write-Host "  * GEOM   - 37M conformers of drug-like molecules with DFT energies."
Write-Host "             https://github.com/learningmatter-mit/geom"
Write-Host "             MIT-licensed data; large (tens of GB). Follow the repo's"
Write-Host "             instructions to fetch the .msgpack archives."
Write-Host "  * CSD torsion library - experimental torsion preferences (ETKDGv3 'ET')."
Write-Host "             https://www.ccdc.cam.ac.uk"
Write-Host "             REQUIRES a CCDC license -- this script will NOT bypass it."
Write-Host "             Register/obtain a licence via the CCDC website."
Write-Host "  * COD    - open crystal structures for torsion validation."
Write-Host "             https://www.crystallography.net"
Write-Host "  * PDB    - small-molecule conformations from deposited structures."
Write-Host "             https://www.rcsb.org"
Write-Host ""
Write-Host "To customize the offline (synthetic) demo parameters instead, run:"
Write-Host "    python scripts/make_synthetic.py --rmsd 1.0 --top 5"
