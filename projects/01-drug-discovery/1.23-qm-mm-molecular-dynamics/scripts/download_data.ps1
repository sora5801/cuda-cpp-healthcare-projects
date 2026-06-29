# ===========================================================================
# scripts/download_data.ps1  --  Real QM/MM data + tool pointers (Windows)
# ---------------------------------------------------------------------------
# Project 1.23 : QM/MM Molecular Dynamics   (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. THERE IS NOTHING TO DOWNLOAD for this demo:
# the ensemble is generated from data/sample/ensemble_params.txt by the program.
# This script just points at the real datasets and production QM/MM engines a
# learner would graduate to.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 1.23 -- QM/MM Molecular Dynamics"
Write-Host ""
Write-Host "There is NO file to download: the program derives every trajectory's"
Write-Host "(field, x0) from the sweep in data/sample/ensemble_params.txt, and the"
Write-Host "model potential-energy surface is built analytically in src/qmmm.h."
Write-Host ""
Write-Host "For REAL QM/MM (enzyme reactions, covalent inhibitors, proton wires):"
Write-Host "  Enzyme-drug complexes (PDB) : https://www.rcsb.org"
Write-Host "  Enzyme reaction database    : https://www.brenda-enzymes.org"
Write-Host "  SAMPL blind-challenge sets  : https://github.com/samplchallenges"
Write-Host ""
Write-Host "Production GPU QM/MM engines to graduate to:"
Write-Host "  AMBER + QUICK (GPU DFT) : https://github.com/merzlab/QUICK"
Write-Host "  TeraChem (GPU DFT)      : https://www.petachem.com"
Write-Host "  OpenMM + PySCF QM/MM    : https://github.com/openmm/openmm"
Write-Host "  CP2K (periodic QM/MM)   : https://github.com/cp2k/cp2k"
Write-Host ""
Write-Host "Bigger SYNTHETIC ensemble (no download):"
Write-Host "  python scripts/make_synthetic.py --nf 64 --nx 64"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
