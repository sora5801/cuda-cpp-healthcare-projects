#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.2 -- Particle-Mesh Ewald Electrostatics
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. The real PME benchmark systems
# require accounts and/or are large; this script prints how to obtain them and
# defers to scripts/make_synthetic.py for the offline demo stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.2 -- Particle-Mesh Ewald Electrostatics"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a TINY committed SYNTHETIC sample (an NaCl-like ionic"
echo "crystal) so the demo runs offline. No download is required to learn PME."
echo
echo "Real periodic MD benchmark systems (for a larger, realistic run):"
echo "  * CHARMM-GUI Archive  -- pre-built solvated protein-water boxes (PSF/PDB)."
echo "      https://charmm-gui.org/?doc=archive   (free account required)"
echo "  * MemProtMD           -- membrane-protein systems in periodic boxes."
echo "      https://memprotmd.bioch.ox.ac.uk/"
echo "  * D. E. Shaw Research Anton trajectories -- ms-scale MD archives."
echo "      Request access from DE Shaw Research (not redistributable here)."
echo
echo "  These formats (PSF/PDB/DCD) carry per-atom charges and box vectors. A real"
echo "  loader would parse charges + coordinates + the periodic box from them; our"
echo "  loader uses a plain '<n> <box>' + 'x y z q' text format (see data/README.md)."
echo "  Respect every dataset's license; none are redistributed in this repo."
echo
echo "  For a larger SYNTHETIC system right now (e.g. an 8x8x8 = 512-ion lattice):"
echo "    python scripts/make_synthetic.py --reps 8 --box 16.0"
echo
echo "[download_data] Nothing to download; synthetic sample already present."
