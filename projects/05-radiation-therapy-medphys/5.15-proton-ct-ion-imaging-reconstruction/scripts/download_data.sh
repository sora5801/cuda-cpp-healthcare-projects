#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL proton-CT data (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 5.15 : Proton CT & Ion Imaging Reconstruction
#
# CONTRACT (CLAUDE.md §8): prints source pointers; downloads NOTHING and NEVER
# bypasses credentials/registration. The committed synthetic sample runs the
# demo offline; scripts/make_synthetic.py generates larger synthetic problems.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.15 -- Proton CT & Ion Imaging Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real / standard proton-CT list-mode data:"
echo "  * TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) or GATE -- simulate a"
echo "    pCT scan and export per-proton entry/exit tracks + residual range,"
echo "    then convert to this project's list-mode format (see data/README.md)."
echo "  * PRaVDA / PRIMA proton-CT consortia -- prototype-scanner datasets"
echo "    (verify current URLs; registration may be required -- this script"
echo "    does NOT bypass it)."
echo "  * ACE collaboration proton-CT phantom datasets (verify URL)."
echo
echo "Offline stand-in (no download, reproducible, SYNTHETIC):"
echo "  python scripts/make_synthetic.py                       # the committed sample"
echo "  python scripts/make_synthetic.py --n 48 --angles 90 --rays 48   # larger"
echo
echo "When wiring a real dataset, keep it idempotent:"
echo "  1) skip the download if the file already exists with the right SHA256"
echo "  2) print source URL + expected size + checksum"
echo "  3) for credentialed sets, print registration instructions ONLY"
