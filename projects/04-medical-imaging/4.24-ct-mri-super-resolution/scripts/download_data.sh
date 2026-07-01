#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.24 : CT/MRI Super-Resolution
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real CT/MRI SR datasets require
# accounts and/or forbid redistribution, so this script prints instructions +
# links ONLY and defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.24 -- CT/MRI Super-Resolution"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample data/sample/phantom_hr.txt (SYNTHETIC) already lets"
echo "the demo run offline. Real datasets below require registration -- this"
echo "script does NOT bypass it; it only prints where to get them."
echo
echo "Real CT/MRI super-resolution datasets (register + accept terms yourself):"
echo "  * HCP 7T/3T paired brain MRI : https://db.humanconnectome.org/"
echo "  * fastMRI                     : https://fastmri.med.nyu.edu/"
echo "  * IXI brain MRI (CC BY-SA 3.0): https://brain-development.org/ixi-dataset/"
echo "  * MSD CT/MRI tasks           : http://medicaldecathlon.com/"
echo
echo "To turn a real slice into this project's input format:"
echo "  1) load one axial slice, normalize intensities to [0,1];"
echo "  2) crop/pad both dims to a multiple of SR_SCALE (=2);"
echo "  3) write '<w> <h>' then w*h floats row-major (see data/README.md)."
echo
echo "For a larger SYNTHETIC phantom instead, run:"
echo "  python scripts/make_synthetic.py --size 128"
