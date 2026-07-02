# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 6.17 -- Purkinje System & Conduction System Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This project ships a reduced-scope
# TEACHING model that runs entirely on the committed synthetic sample, so there
# is no required download. This script points to the real research datasets and
# defers to scripts/make_synthetic.py for the offline stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.17 -- Purkinje System & Conduction System Modeling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project runs on the committed SYNTHETIC sample (data/sample/purkinje_tree.txt)."
Write-Host "No download is required to build, run, or verify the demo."
Write-Host ""
Write-Host "To regenerate / enlarge the synthetic tree:"
Write-Host "    python scripts/make_synthetic.py"
Write-Host ""
Write-Host "Real Purkinje-network geometries and His-bundle electrograms (study these to"
Write-Host "extend the project -- see README 'Exercises'):"
Write-Host "  * openCARP community Purkinje experiments:"
Write-Host "      https://opencarp.org/community/community-experiments"
Write-Host "  * MonoAlg3D_C Purkinje examples (GPU monodomain + PMJ calibration):"
Write-Host "      https://github.com/rsachetto/MonoAlg3D_C"
Write-Host "  * NeuroMorpho (branching-tree morphologies, analogy):"
Write-Host "      https://neuromorpho.org"
Write-Host "  * PhysioNet His-bundle electrogram databases (may require registration):"
Write-Host "      https://physionet.org"
Write-Host ""
Write-Host "NOTE: PhysioNet and similar sources may require an account + a signed data-use"
Write-Host "agreement. This script prints instructions ONLY and never bypasses that step."
