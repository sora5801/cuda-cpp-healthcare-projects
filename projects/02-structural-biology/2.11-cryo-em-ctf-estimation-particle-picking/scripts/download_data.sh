#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.11 : Cryo-EM CTF Estimation & Particle Picking
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size, and NEVER bypasses credentials/registration. Real cryo-EM micrographs
# live in EMPIAR and are large (tens of GB) binary MRC files; converting them to
# this project's tiny text format is out of scope for a teaching demo, so this
# script PRINTS guidance and defers to scripts/make_synthetic.py for an offline,
# verifiable stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.11 -- Cryo-EM CTF Estimation & Particle Picking"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real cryo-EM micrograph datasets (free, but large; MRC binary format):"
echo "  * EMPIAR archive home : https://www.ebi.ac.uk/empiar/"
echo "  * EMPIAR-10025 (beta-galactosidase) : classic CTF/processing tutorial set"
echo "  * EMPIAR-10064 (T20S proteasome)    : RELION tutorial micrographs"
echo "  * RELION tutorial data : https://relion.readthedocs.io"
echo
echo "These are tens of GB and need an MRC reader (e.g. Python 'mrcfile') to"
echo "convert a micrograph into this project's text format:"
echo "    line 1:  n pixel_size lambda cs amp_contrast true_dz"
echo "    body  :  n*n floats (row-major)"
echo "(set true_dz = -1 for real data, where the defocus is unknown.)"
echo
echo "The committed tiny SYNTHETIC sample in data/sample/ is enough to run the"
echo "demo offline. For a larger synthetic problem, run:"
echo "    python scripts/make_synthetic.py --n 256 --defocus 12000"
