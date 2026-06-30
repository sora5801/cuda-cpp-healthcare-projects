# ===========================================================================
# scripts/download_data.ps1  --  Real cryo-EM data pointers (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.12 : Flexible Fitting / MDFF
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. There is NOTHING to download for the
# demo -- the committed sample is a self-contained synthetic problem. This script
# only points at the real public maps/structures a learner would fit for real.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.12 -- Flexible Fitting / MDFF"
Write-Host ""
Write-Host "There is no file to download: the demo runs on a self-contained"
Write-Host "synthetic problem in data/sample/mdff_problem.txt (atoms + parameters;"
Write-Host "the density grid is rebuilt by the program)."
Write-Host ""
Write-Host "For REAL cryo-EM flexible fitting you need a density MAP + a starting MODEL:"
Write-Host "  EMDB   : https://www.ebi.ac.uk/emdb/    (reference density maps, MRC/CCP4)"
Write-Host "  EMPIAR : https://www.ebi.ac.uk/empiar/  (raw particle data)"
Write-Host "  PDB    : ribosome MDFF benchmarks 3J7Y, 4V6X  (https://www.rcsb.org)"
Write-Host "  Tools  : NAMD/VMD MDFF (https://www.ks.uiuc.edu/Research/namd/),"
Write-Host "           phenix.real_space_refine, Coot."
Write-Host ""
Write-Host "Wiring a real map would add an MRC/CCP4 reader (-> rho) and a PDB reader"
Write-Host "(-> x0); the fitting kernel itself is unchanged."
Write-Host ""
Write-Host "Larger SYNTHETIC problem (no download):"
Write-Host "  python scripts/make_synthetic.py --iters 400"
Write-Host ""
Write-Host "Target data dir: $DataDir"
