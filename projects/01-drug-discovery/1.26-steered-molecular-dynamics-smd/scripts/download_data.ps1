# ===========================================================================
# scripts/download_data.ps1  --  SMD data pointers (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.26 : Steered Molecular Dynamics (SMD)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. For THIS project there is nothing to fetch -- the
# reduced 1-D model is fully specified by the 14 numbers in
# data/sample/smd_config.txt. This script prints where to find REAL full-atom SMD
# material and defers to scripts/make_synthetic.py for larger offline ensembles.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.26 -- Steered Molecular Dynamics (SMD)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "There is NO file to download: the 1-D teaching model is fully defined by"
Write-Host "data/sample/smd_config.txt (14 numbers; see data/README.md)."
Write-Host ""
Write-Host "For REAL full-atom SMD (pull a ligand out of a pocket in a true MD field):"
Write-Host "  NAMD SMD tutorials : https://www.ks.uiuc.edu/Training/Tutorials/"
Write-Host "                       (constant-velocity / constant-force SMD walkthroughs)"
Write-Host "  GROMACS pull code  : https://github.com/gromacs/gromacs   (GPU pull-coord)"
Write-Host "  OpenMM             : https://github.com/openmm/openmm     (CustomExternalForce)"
Write-Host "  alchemlyb          : https://github.com/alchemistry/alchemlyb (Jarzynski post-proc)"
Write-Host "  BindingDB          : https://www.bindingdb.org           (residence-time data)"
Write-Host "  PDB                : https://www.rcsb.org                 (force-probe structures)"
Write-Host ""
Write-Host "Bigger SYNTHETIC ensemble for this project (no download, tighter Jarzynski):"
Write-Host "  python scripts/make_synthetic.py --n-traj 65536"
