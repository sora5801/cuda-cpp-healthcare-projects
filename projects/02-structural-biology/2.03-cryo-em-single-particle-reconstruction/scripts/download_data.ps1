# ===========================================================================
# scripts/download_data.ps1  --  Pointers to the FULL datasets (Windows)
# ---------------------------------------------------------------------------
# Project 2.3 : Cryo-EM Single-Particle Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials / data-use agreements. Real cryo-EM particle stacks
# (EMPIAR) are large and some require acknowledging a license; this script only
# prints instructions + links and defers to scripts/make_synthetic.py for an
# offline stand-in. The committed sample already runs the demo with no download.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.3 -- Cryo-EM Single-Particle Reconstruction"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This teaching project runs on a SYNTHETIC 2-D sample (data/sample/);"
Write-Host "no download is required. To explore REAL cryo-EM data:"
Write-Host ""
Write-Host "  EMDB    3-D density maps (MRC/.map)   https://www.ebi.ac.uk/emdb/"
Write-Host "  EMPIAR  raw particle image stacks      https://www.ebi.ac.uk/empiar/"
Write-Host "  RCSB    atomic models fit into maps    https://www.rcsb.org"
Write-Host "  cryoDRGN benchmark datasets            https://github.com/ml-struct-bio/cryodrgn"
Write-Host ""
Write-Host "NOTE: EMPIAR entries are tens of GB and some require accepting a"
Write-Host "      data-use agreement. This script does NOT bypass that -- follow"
Write-Host "      the entry's instructions on the EMPIAR website to download."
Write-Host ""
Write-Host "For a larger SYNTHETIC problem (any size, fully offline), run:"
Write-Host "    python scripts/make_synthetic.py --n 100000"
