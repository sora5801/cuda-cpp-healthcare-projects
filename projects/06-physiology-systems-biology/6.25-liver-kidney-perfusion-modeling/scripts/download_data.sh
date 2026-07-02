#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.25 : Liver & Kidney Perfusion Modeling
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URL +
# expected size + checksum, and NEVER bypasses credentials/registration. The
# committed synthetic sample already runs the demo; the real sources below feed a
# richer, calibrated lobule/nephron model (an exercise in the README).
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.25 -- Liver & Kidney Perfusion Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC lobule config (data/sample/lobule.txt), so no"
echo "download is required to run the demo. To build a physiologically calibrated"
echo "model, curate these PUBLIC sources into the loader's field layout:"
echo
echo "  * Human Protein Atlas -- liver zonal enzyme expression (Vmax gradient):"
echo "      https://www.proteinatlas.org   (CC BY-SA 3.0; browse per-enzyme liver data)"
echo "  * HMDB -- liver metabolite concentrations (set C_in / Km scales):"
echo "      https://hmdb.ca                 (free for academic use; see terms)"
echo "  * Open Systems Pharmacology PBPK model library -- organ clearance params:"
echo "      https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library  (GPLv2)"
echo "  * PhysioNet -- renal function datasets (credentialed for some sets):"
echo "      https://physionet.org           (register; this script will NOT bypass it)"
echo
echo "  For a larger SYNTHETIC problem instead, run:"
echo "    python scripts/make_synthetic.py --nsin 1048576"
echo
echo "  Idempotent pattern when wiring a real fetch:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
