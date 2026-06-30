#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.8 -- GPU Molecular Visualization & Ray Tracing
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This project renders the committed SYNTHETIC
# sample by default; real structures (PDB/EMDB) are large and carry their own
# per-entry terms, and are NOT required for the demo. So this script only prints
# pointers and defers to scripts/make_synthetic.py for offline data.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.8 -- GPU Molecular Visualization & Ray Tracing"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The demo runs entirely on the committed SYNTHETIC sample:"
echo "    data/sample/molecule_sample.scene   (no download needed)"
echo "Regenerate or resize it with:"
echo "    python scripts/make_synthetic.py --turns 6 --width 320 --height 320"
echo
echo "To render a REAL structure, fetch one from these sources (check each"
echo "entry's license) and convert it to the .scene format (see data/README.md):"
echo "  - RCSB PDB (atoms):         https://www.rcsb.org"
echo "  - EMDB (cryo-EM volumes):   https://www.ebi.ac.uk/emdb/"
echo "  - GPCRmd (MD trajectories): https://gpcrmd.org"
echo "  - CHARMM-GUI (systems):     https://charmm-gui.org"
echo
echo "Example (PDB, public): download a structure by id, e.g."
echo "    curl -L -o data/1ubq.pdb https://files.rcsb.org/download/1UBQ.pdb"
echo "then write a small converter (Exercise in README.md) PDB -> .scene."
echo "This script downloads nothing automatically by design."
