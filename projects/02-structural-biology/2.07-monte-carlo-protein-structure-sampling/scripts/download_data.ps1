# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.7 : Monte Carlo Protein Structure Sampling (HP lattice model)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This reduced-scope teaching model
# folds a SYNTHETIC HP sequence, so no external download is required to run the
# demo. The links below point to the real-world benchmarks a learner would use
# if they extended this toward a full-atom MC engine.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.7 -- Monte Carlo Protein Structure Sampling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This reduced-scope HP-lattice demo needs NO download: data/sample/hp_problem.txt"
Write-Host "is a tiny SYNTHETIC HP sequence and the program folds it offline."
Write-Host ""
Write-Host "For a different SYNTHETIC problem (longer chain, more replicas), run:"
Write-Host "    python scripts/make_synthetic.py --sequence HHPPHPPHPPHPPHPPHPPHPPHH --replicas 1024"
Write-Host ""
Write-Host "Real-world folding/sampling benchmarks (study these to go further):"
Write-Host "  * CASP structure-prediction benchmarks : https://predictioncenter.org"
Write-Host "  * PDB experimental structures          : https://www.rcsb.org"
Write-Host "  * Dunbrack backbone-dependent rotamers : https://dunbrack.fccc.edu/bbdep2010/"
Write-Host "  * CAMEO continuous benchmarking        : https://www.cameo3d.org"
Write-Host ""
Write-Host "These are full 3-D coordinate / rotamer datasets that a production MC engine"
Write-Host "(Rosetta, OpenMM) consumes; this 2-D HP teaching model does not parse them."
Write-Host "Respect each site's license/registration -- this script never bypasses it."
