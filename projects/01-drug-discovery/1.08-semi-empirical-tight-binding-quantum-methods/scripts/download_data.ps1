# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.8 -- Semi-Empirical & Tight-Binding Quantum Methods
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. The real datasets
# for this method are large molecular sets with 3-D geometries; turning them into
# pi-system graphs needs a chemistry toolkit (RDKit / Open Babel) and is out of
# scope for this teaching project, which consumes connectivity graphs directly.
# This script therefore prints guidance and links only; the committed synthetic
# sample (or make_synthetic.py) is what the demo runs on.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.8 -- Semi-Empirical & Tight-Binding Quantum Methods"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project runs on a tiny SYNTHETIC sample (data/sample/molecules_sample.txt)"
Write-Host "and needs NO download. The real datasets the method is benchmarked on are:"
Write-Host ""
Write-Host "  ANI-1   ~20M DFT energies on 57k molecules   https://github.com/isayev/ANI1   (CC0)"
Write-Host "  QM9     134k molecules + DFT properties       https://doi.org/10.6084/m9.figshare.978904"
Write-Host "  GMTKN55 thermochemistry/kinetics benchmark    https://www.chemie.uni-bonn.de/grimme/de/software/gmtkn"
Write-Host "  COMPAS  polycyclic aromatic systems           (verify current URL)"
Write-Host ""
Write-Host "These ship 3-D geometries; convert each molecule to a pi-system graph with a chemistry"
Write-Host "toolkit (RDKit/Open Babel) to feed this project's loader. None require bypassing any"
Write-Host "credentials; respect each dataset's license."
Write-Host ""
Write-Host "To (re)generate the committed synthetic batch instead, run:"
Write-Host "    python scripts/make_synthetic.py"
