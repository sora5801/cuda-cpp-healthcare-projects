# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 2.8 -- GPU Molecular Visualization & Ray Tracing
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL, and
# NEVER bypasses credentials/registration. This project renders the committed
# SYNTHETIC sample by default; real structures (PDB/EMDB) are large, carry their
# own per-entry terms, and are NOT required for the demo. So this script only
# prints pointers and defers to scripts/make_synthetic.py for offline data.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 2.8 -- GPU Molecular Visualization & Ray Tracing"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "The demo runs entirely on the committed SYNTHETIC sample:"
Write-Host "    data/sample/molecule_sample.scene   (no download needed)"
Write-Host "Regenerate or resize it with:"
Write-Host "    python scripts/make_synthetic.py --turns 6 --width 320 --height 320"
Write-Host ""
Write-Host "To render a REAL structure, fetch one from these sources (check each"
Write-Host "entry's license) and convert it to the .scene format (see data/README.md):"
Write-Host "  - RCSB PDB (atoms):        https://www.rcsb.org"
Write-Host "  - EMDB (cryo-EM volumes):  https://www.ebi.ac.uk/emdb/"
Write-Host "  - GPCRmd (MD trajectories):https://gpcrmd.org"
Write-Host "  - CHARMM-GUI (systems):    https://charmm-gui.org"
Write-Host ""
Write-Host "Example (PDB, public): download a structure by id with curl, e.g."
Write-Host "    curl -L -o data/1ubq.pdb https://files.rcsb.org/download/1UBQ.pdb"
Write-Host "then write a small converter (Exercise in README.md) PDB -> .scene."
Write-Host "This script downloads nothing automatically by design."
