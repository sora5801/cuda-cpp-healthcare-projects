#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Realistic MARTINI-system pointers (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 2.5 : Coarse-Grained / MARTINI Simulation. Nothing to download.
#
# CONTRACT (CLAUDE.md §8): the committed sample is synthetic, so there is no file
# to fetch. This script only prints where REAL MARTINI systems come from and
# never bypasses any registration. For a bigger run, use make_synthetic.py.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 2.5 -- Coarse-Grained / MARTINI Simulation"
echo
echo "There is no file to download: data/sample/cg_system.txt is synthetic"
echo "and self-contained (scripts/make_synthetic.py)."
echo
echo "For REAL MARTINI systems and production CG-MD:"
echo "  CHARMM-GUI Martini Maker : https://charmm-gui.org   (membrane builder; registration)"
echo "  MARTINI force field      : https://cgmartini.nl     (official bead types + eps matrix)"
echo "  insane.py                : https://github.com/Tsjerk/Insane   (bilayer assembly)"
echo "  TS2CG                    : https://github.com/weria-pezeshkian/TS2CG"
echo "  GROMACS                  : https://github.com/gromacs/gromacs (GPU CG-MD engine)"
echo "  EMDB (validation maps)   : https://www.ebi.ac.uk/emdb/"
echo
echo "Bigger SYNTHETIC system (no download):"
echo "  python scripts/make_synthetic.py --per-side 4 --steps 600"
echo
echo "Target data dir: $PROJECT_ROOT/data"
