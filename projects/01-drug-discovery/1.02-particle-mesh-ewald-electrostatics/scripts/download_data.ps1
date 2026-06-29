# ===========================================================================
# scripts/download_data.ps1  --  Fetch the FULL dataset (Windows / PowerShell)
# ---------------------------------------------------------------------------
# Project 1.2 -- Particle-Mesh Ewald Electrostatics
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# expected size, and NEVER bypasses credentials/registration. The real PME
# benchmark systems (CHARMM-GUI boxes, MemProtMD membranes, Anton trajectories)
# require registration and/or are large; this script only prints how to obtain
# them and defers to scripts/make_synthetic.py for the offline demo stand-in.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"

Write-Host "[download_data] Project 1.2 -- Particle-Mesh Ewald Electrostatics"
Write-Host "[download_data] Target data dir: $DataDir"
Write-Host ""
Write-Host "This project ships a TINY committed SYNTHETIC sample (an NaCl-like ionic"
Write-Host "crystal) so the demo runs offline. No download is required to learn PME."
Write-Host ""
Write-Host "Real periodic MD benchmark systems (for a larger, realistic run):"
Write-Host "  * CHARMM-GUI Archive  -- pre-built solvated protein-water boxes (PSF/PDB)."
Write-Host "      https://charmm-gui.org/?doc=archive   (free account required)"
Write-Host "  * MemProtMD           -- membrane-protein systems in periodic boxes."
Write-Host "      https://memprotmd.bioch.ox.ac.uk/"
Write-Host "  * D. E. Shaw Research Anton trajectories -- ms-scale MD archives."
Write-Host "      Request access from DE Shaw Research (not redistributable here)."
Write-Host ""
Write-Host "  These formats (PSF/PDB/DCD) carry per-atom charges and box vectors. A real"
Write-Host "  loader would parse charges + coordinates + the periodic box from them; our"
Write-Host "  loader uses a plain '<n> <box>' + 'x y z q' text format (see data/README.md)."
Write-Host "  Respect every dataset's license; none are redistributed in this repo."
Write-Host ""
Write-Host "  For a larger SYNTHETIC system right now (e.g. an 8x8x8 = 512-ion lattice):"
Write-Host "    python scripts/make_synthetic.py --reps 8 --box 16.0"
Write-Host ""
Write-Host "[download_data] Nothing to download; synthetic sample already present."
