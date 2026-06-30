# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.1 : Protein Structure Prediction Inference (AlphaFold-class)
#               REDUCED-SCOPE TEACHING VERSION.
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This teaching project runs on a tiny
# SYNTHETIC sample (data/sample/attention_sample.txt); no real dataset is needed
# to build or demo it. This script therefore only prints where the real data
# lives so a curious learner can go further, and defers to make_synthetic.py for
# a bigger offline problem.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.1 -- Protein Structure Prediction Inference (AlphaFold-class)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project needs NO download: the committed synthetic sample in"
Write-Host "  data/sample/attention_sample.txt"
Write-Host "is sufficient to build, run, and verify the demo offline."
Write-Host ""
Write-Host "Where the real-world data lives (for further study, NOT required here):"
Write-Host "  * AlphaFold Protein Structure DB (200M+ predicted structures):"
Write-Host "      https://alphafold.ebi.ac.uk/"
Write-Host "  * RCSB PDB (227k+ experimental structures):  https://www.rcsb.org"
Write-Host "  * UniProt / UniRef90 (MSA sequence databases): https://www.uniprot.org"
Write-Host "  * CAMEO / CASP15 prediction benchmarks:        https://www.cameo3d.org"
Write-Host ""
Write-Host "Note: a real AlphaFold/ESMFold run also needs multi-gigabyte trained"
Write-Host "model WEIGHTS and (for AF2) MSA databases -- see those projects' repos."
Write-Host "This teaching version uses random synthetic Q/K/V instead (no weights)."
Write-Host ""
Write-Host "For a larger SYNTHETIC attention problem (more residues), run:"
Write-Host "    python scripts/make_synthetic.py --L 64"
