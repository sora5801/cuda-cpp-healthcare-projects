# ===========================================================================
# scripts/download_data.ps1  --  Real-structure PBE input pointers (Windows)
# ---------------------------------------------------------------------------
# Project 2.9 : Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
#
# CONTRACT (CLAUDE.md sec.8): idempotent, documented, prints sources, and NEVER
# bypasses credentials. There is NOTHING to download for the demo: the committed
# data/sample/molecule.pqr (synthetic) is enough. This script tells you how to
# get a REAL protein into the same .pqr-style input format.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.9 -- Poisson-Boltzmann Electrostatics"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "Nothing to download: the demo runs on the committed SYNTHETIC sample"
Write-Host "  data/sample/molecule.pqr  (a tiny dipolar 'molecule')."
Write-Host ""
Write-Host "To run on a REAL protein, produce a .pqr (atoms with partial charge +"
Write-Host "radius) and convert it to this project's 1-line-header format:"
Write-Host "  1) Fetch a structure from the RCSB PDB:    https://www.rcsb.org"
Write-Host "  2) Add charges + radii with PDB2PQR:       https://github.com/Electrostatics/pdb2pqr"
Write-Host "       pdb2pqr30 --ff=AMBER 1abc.pdb 1abc.pqr"
Write-Host "  3) Reformat the ATOM lines (columns x y z q radius) into our file:"
Write-Host "       header: 'natoms n h eps_in eps_out kappa2 iters' then one"
Write-Host "       'x y z q radius' line per atom  (see data/README.md)."
Write-Host ""
Write-Host "Reference solvers / benchmarks for comparison:"
Write-Host "  APBS    : https://github.com/Electrostatics/apbs   (PB solver + tests)"
Write-Host "  DelPhi  : http://compbio.clemson.edu/delphi"
Write-Host "  OpenMM  : https://github.com/openmm/openmm         (GPU Generalized Born)"
Write-Host ""
Write-Host "Bigger synthetic problem (no download):"
Write-Host "  python scripts/make_synthetic.py --n 64 --iters 800"
