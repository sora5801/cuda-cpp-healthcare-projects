# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.15 : Antibody Structure Prediction  (reduced-scope: CDR screening)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real antibody databases each have
# their own access terms, so this script PRINTS INSTRUCTIONS ONLY and points at
# scripts/make_synthetic.py for the offline stand-in the demo actually uses.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.15 -- Antibody Structure Prediction (CDR screening)"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project ships a SYNTHETIC sample (data/sample/antibodies_sample.txt)"
Write-Host "so the demo runs offline. To work with REAL antibody CDR sequences/structures,"
Write-Host "obtain them from the sources below and convert to the loader's text format"
Write-Host "(see data/README.md). Each source has its own license -- respect it."
Write-Host ""
Write-Host "  SAbDab (Structural Antibody Database) -- IMGT-numbered antibody structures:"
Write-Host "    https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/"
Write-Host "  Thera-SAbDab (therapeutic antibodies):"
Write-Host "    https://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/"
Write-Host "  OAS (Observed Antibody Space) -- ~2 billion antibody sequences:"
Write-Host "    https://opig.stats.ox.ac.uk/webapps/oas/"
Write-Host ""
Write-Host "  To convert a real set to the screen's format you would:"
Write-Host "    1) IMGT-number each Fv (e.g. with ANARCI) to delimit the six CDR loops,"
Write-Host "    2) emit one line per antibody: '<name> H1 H2 H3 L1 L2 L3' (amino-acid strings),"
Write-Host "    3) put one antibody on a 'QUERY <name> ...' line to screen the rest against."
Write-Host ""
Write-Host "  For a larger SYNTHETIC problem right now (no download needed):"
Write-Host "    python scripts/make_synthetic.py --n 1048576"
