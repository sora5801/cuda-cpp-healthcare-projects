# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.5 : Free Energy Perturbation / Thermodynamic Integration
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This project is a REDUCED-SCOPE teaching
# model with no real dataset to fetch -- a faithful FEP/TI run needs a full MD
# engine + force field. So this script only prints links and defers to the
# committed synthetic sample (scripts/make_synthetic.py) for an offline run.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.5 -- Free Energy Perturbation / Thermodynamic Integration"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This is a REDUCED-SCOPE teaching model: it samples a 1-D harmonic"
Write-Host "alchemical system whose DeltaG has a CLOSED FORM, so no external data"
Write-Host "is required. The committed synthetic sample runs the demo offline."
Write-Host ""
Write-Host "Real FEP/TI benchmarks to study (each needs a full MD engine):"
Write-Host "  * Merck FEP benchmark set (open, via OpenFE):"
Write-Host "      https://github.com/OpenFreeEnergy/openfe"
Write-Host "  * FEP+ validation set (Schrodinger; registration required) -- links only."
Write-Host "  * PDBbind experimental binding affinities:  http://www.pdbbind.org.cn"
Write-Host "  * ChEMBL bioactivity data:                  https://www.ebi.ac.uk/chembl/"
Write-Host ""
Write-Host "For a different SYNTHETIC problem, run:"
Write-Host "    python scripts/make_synthetic.py --kB 9 --windows 21"
Write-Host ""
Write-Host "NOTE: credentialed sets (FEP+, some PDBbind tiers) require you to"
Write-Host "register and accept their license yourself; this script will never"
Write-Host "attempt to bypass that."
