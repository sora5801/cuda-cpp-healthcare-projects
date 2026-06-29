#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to the real datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.14 : Conformer Ensemble Generation
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + access
# notes, and NEVER bypasses credentials/registration.
#
# IMPORTANT: this teaching demo is SELF-CONTAINED -- the molecule is fixed in
# src/conformer.h and the committed data/sample/conformer_params.txt is all the
# demo needs. There is NOTHING to download to run this project. The datasets
# below are what you would use to VALIDATE a production conformer generator.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.14 -- Conformer Ensemble Generation"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project needs NO download to run: data/sample/conformer_params.txt"
echo "plus the compile-time molecule in src/conformer.h are sufficient."
echo
echo "Real-world reference datasets (for validating a production generator):"
echo "  * GEOM   - 37M conformers of drug-like molecules with DFT energies."
echo "             https://github.com/learningmatter-mit/geom"
echo "             MIT-licensed data; large (tens of GB). Follow the repo's"
echo "             instructions to fetch the .msgpack archives."
echo "  * CSD torsion library - experimental torsion preferences (ETKDGv3 'ET')."
echo "             https://www.ccdc.cam.ac.uk"
echo "             REQUIRES a CCDC license -- this script will NOT bypass it."
echo "             Register/obtain a licence via the CCDC website."
echo "  * COD    - open crystal structures for torsion validation."
echo "             https://www.crystallography.net"
echo "  * PDB    - small-molecule conformations from deposited structures."
echo "             https://www.rcsb.org"
echo
echo "To customize the offline (synthetic) demo parameters instead, run:"
echo "    python scripts/make_synthetic.py --rmsd 1.0 --top 5"
