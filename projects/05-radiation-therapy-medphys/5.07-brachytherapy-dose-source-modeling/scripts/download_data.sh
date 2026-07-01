#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Point at the FULL datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 5.7 : Brachytherapy Dose & Source Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. The real TG-43 consensus source
# datasets live in published journal tables (below); we do not redistribute
# them. The committed synthetic sample runs the demo offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.7 : Brachytherapy Dose & Source Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC plan in data/sample/plan_sample.txt;"
echo "no download is required to run the demo. Real TG-43 datasets:"
echo
echo "  * AAPM TG-43U1 consensus source data (radial dose g_L(r) and anisotropy"
echo "    F(r,theta) tables per source model, e.g. Ir-192 HDR, Pd-103, I-125):"
echo "    https://www.aapm.org/pubs/reports/"
echo "  * ESTRO ACROP brachytherapy guideline test cases (planning geometry)."
echo "  * TCIA prostate brachytherapy CT datasets (imaging; free registration):"
echo "    https://www.cancerimagingarchive.net/"
echo
echo "To transcribe a real source's tables into this project's plan format,"
echo "edit data/sample/plan_sample.txt (format documented in data/README.md)."
echo "For a larger SYNTHETIC grid, run:"
echo "    python scripts/make_synthetic.py --grid 81"
