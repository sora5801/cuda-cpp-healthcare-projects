# ===========================================================================
# scripts/download_data.ps1  --  Pointers to REAL metabolic models (Windows)
# ---------------------------------------------------------------------------
# Project 6.12 : Metabolic Flux / Constraint-Based Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL, and
# NEVER bypasses credentials. The committed sample is a tiny SYNTHETIC toy model
# (data/sample/toy_core_model.txt) that runs the demo offline; the genome-scale
# models below are public but big and in SBML/JSON, which our simple text loader
# does not parse -- so this script only prints where to get them and how a real
# workflow (COBRApy) would consume them.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 6.12 -- Metabolic Flux / Constraint-Based Modeling"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a SYNTHETIC toy model (data/sample/toy_core_model.txt)."
Write-Host "It runs the demo with zero downloads. To regenerate or resize it:"
Write-Host "    python scripts/make_synthetic.py"
Write-Host ""
Write-Host "Real genome-scale metabolic models (public, but SBML/JSON -- not our"
Write-Host "text format; use COBRApy to read them):"
Write-Host "  * BiGG Models (curated GEMs):     http://bigg.ucsd.edu/models"
Write-Host "      e.g. E. coli core (95 rxns):  http://bigg.ucsd.edu/models/e_coli_core"
Write-Host "  * Recon3D (human, ~10600 rxns):   https://github.com/SBRG/Recon3D"
Write-Host "  * Virtual Metabolic Human portal: https://vmh.life"
Write-Host ""
Write-Host "Typical real workflow (outside this teaching repo):"
Write-Host "    pip install cobra"
Write-Host "    python -c \""import cobra; m=cobra.io.load_model('e_coli_core'); print(m.optimize())\"""
Write-Host ""
Write-Host "See THEORY.md 'Where this sits in the real world' for how production FBA"
Write-Host "differs (sparse interior-point / revised simplex over 1000s of reactions)."
