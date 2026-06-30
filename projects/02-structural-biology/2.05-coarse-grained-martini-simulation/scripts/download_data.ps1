# ===========================================================================
# scripts/download_data.ps1  --  Realistic MARTINI-system pointers (Windows)
# ---------------------------------------------------------------------------
# Project 2.5 : Coarse-Grained / MARTINI Simulation. Nothing to download.
#
# CONTRACT (CLAUDE.md §8): the committed sample is synthetic, so there is no file
# to fetch. This script only prints where REAL MARTINI systems come from and
# never bypasses any registration. For a bigger run, use make_synthetic.py.
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 2.5 -- Coarse-Grained / MARTINI Simulation"
Write-Host ""
Write-Host "There is no file to download: data/sample/cg_system.txt is synthetic"
Write-Host "and self-contained (scripts/make_synthetic.py)."
Write-Host ""
Write-Host "For REAL MARTINI systems and production CG-MD:"
Write-Host "  CHARMM-GUI Martini Maker : https://charmm-gui.org   (membrane builder; registration)"
Write-Host "  MARTINI force field      : https://cgmartini.nl     (official bead types + eps matrix)"
Write-Host "  insane.py                : https://github.com/Tsjerk/Insane   (bilayer assembly)"
Write-Host "  TS2CG                    : https://github.com/weria-pezeshkian/TS2CG"
Write-Host "  GROMACS                  : https://github.com/gromacs/gromacs (GPU CG-MD engine)"
Write-Host "  EMDB (validation maps)   : https://www.ebi.ac.uk/emdb/"
Write-Host ""
Write-Host "Bigger SYNTHETIC system (no download):"
Write-Host "  python scripts/make_synthetic.py --per-side 4 --steps 600"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
