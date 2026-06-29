# ===========================================================================
# scripts/download_data.ps1  --  Real hydration-free-energy data pointers (Win)
# ---------------------------------------------------------------------------
# Project 1.32 : Alchemical Hydration Free Energy (delta-G_solv)
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URLs and
# NEVER bypasses credentials/registration. This project's demo needs NOTHING to
# download -- the committed sample fully specifies a reproducible calculation, and
# the model bath is generated deterministically. This script only points you at
# the REAL experimental benchmarks you would validate a production engine against.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 1.32 -- Alchemical Hydration Free Energy (dG_solv)"
Write-Host ""
Write-Host "Nothing to fetch: data/sample/alchemy_config.txt fully specifies the"
Write-Host "calculation, and the solvent bath is built deterministically in code."
Write-Host ""
Write-Host "REAL experimental hydration-free-energy benchmarks (for validating a"
Write-Host "production FEP engine -- NOT consumed by this teaching demo):"
Write-Host "  FreeSolv   : https://github.com/MobleyLab/FreeSolv   (643 dG_hyd; permissive)"
Write-Host "  MNSol      : https://comp.chem.umn.edu/mnsol/        (license acceptance required)"
Write-Host "  SAMPL      : https://github.com/samplchallenges/SAMPL (blind challenges)"
Write-Host "  NIST ThermoML : https://trc.nist.gov                 (curated thermochemistry)"
Write-Host ""
Write-Host "MNSol requires accepting a license on its website; this script does NOT"
Write-Host "bypass that -- download it manually if you accept the terms."
Write-Host ""
Write-Host "Bigger / finer SYNTHETIC problem (no download):"
Write-Host "  python scripts/make_synthetic.py --n-windows 21 --n-walkers 256"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
