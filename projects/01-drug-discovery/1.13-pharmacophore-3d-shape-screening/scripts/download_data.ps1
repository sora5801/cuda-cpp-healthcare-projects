# ===========================================================================
# scripts/download_data.ps1  --  Fetch / point to the FULL dataset (Windows)
# ---------------------------------------------------------------------------
# Project 1.13 : Pharmacophore & 3D Shape Screening
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real 3D conformer libraries are
# large and/or licensed and come as SDF/MOL2 (not this project's simple text
# format), so this script PRINTS instructions + links rather than blindly
# downloading gigabytes; for an offline run, the committed synthetic sample (or
# make_synthetic.py) is sufficient.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.13 -- Pharmacophore & 3D Shape Screening"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a tiny SYNTHETIC sample in data/sample/ that is"
Write-Host "enough to build and run the demo offline. No download is required."
Write-Host ""
Write-Host "To screen REAL molecules, obtain 3D conformers from a public library:"
Write-Host "  * ZINC20 conformers : https://zinc20.docking.org   (free for research)"
Write-Host "  * DUD-E             : https://dude.docking.org      (actives + decoys, 3D)"
Write-Host "  * Enamine REAL      : https://enamine.net           (make-on-demand library)"
Write-Host ""
Write-Host "Those come as SDF/MOL2. Convert to this project's 'x y z radius' text"
Write-Host "format with a short RDKit script (read 3D coordinates, map each element"
Write-Host "to its van der Waals radius), then run:"
Write-Host "    .\build\x64\Release\pharmacophore-3d-shape-screening.exe <your_file.txt>"
Write-Host ""
Write-Host "For a larger SYNTHETIC problem with no downloads, regenerate the sample:"
Write-Host "    python scripts\make_synthetic.py"
Write-Host ""
Write-Host "Respect each source's license; none are redistributed in this repo."
