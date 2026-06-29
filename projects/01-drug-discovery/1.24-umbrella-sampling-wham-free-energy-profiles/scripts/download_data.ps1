# ===========================================================================
# scripts/download_data.ps1  --  Real umbrella-sampling pointers (Windows)
# ---------------------------------------------------------------------------
# Project 1.24 : Umbrella Sampling / WHAM Free Energy Profiles
#
# There is NOTHING to download for this project: the "data" is the synthetic
# experiment configuration in data/sample/umbrella.txt, which the program turns
# into biased trajectories on the fly. This script just points at the real-world
# datasets/tools and never bypasses any registration (CLAUDE.md §8).
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 1.24 -- Umbrella Sampling / WHAM Free Energy Profiles"
Write-Host ""
Write-Host "There is no file to download: the program derives every window's"
Write-Host "biased trajectory from data/sample/umbrella.txt (a synthetic double-well)."
Write-Host ""
Write-Host "For REAL umbrella sampling (all-atom MD per window, then WHAM):"
Write-Host "  GROMACS tutorial : https://tutorials.gromacs.org        (gmx wham worked example)"
Write-Host "  SAMPL challenges : https://github.com/samplchallenges/SAMPL  (binding free energy)"
Write-Host "  BindingDB        : https://www.bindingdb.org           (measured affinities)"
Write-Host "  PLUMED           : https://github.com/plumed/plumed2    (collective variables + restraints)"
Write-Host ""
Write-Host "Bigger SYNTHETIC experiment (no download):"
Write-Host "  python scripts/make_synthetic.py --n-windows 51 --n-sample 200000"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
