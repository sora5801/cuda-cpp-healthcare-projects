#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Point at REAL ASL datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.23 : Arterial Spin Labeling & Perfusion Imaging
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. The real ASL datasets below are credentialed or
# large, so this script only PRINTS where to get them and how; the committed
# synthetic sample (scripts/make_synthetic.py) is what the demo actually runs.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.23 -- Arterial Spin Labeling & Perfusion Imaging"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The demo runs on the committed SYNTHETIC sample (data/sample/asl_sample.txt),"
echo "so no download is required. For real multi-delay ASL data, see:"
echo
echo "  1) OpenNeuro ASL datasets (open, BIDS-formatted; search 'ASL'):"
echo "       https://openneuro.org/"
echo "     Many are directly downloadable (no credentials). Pick a multi-PLD/"
echo "     multi-delay pCASL dataset; use the perf/ delta-M series + the PLD list."
echo
echo "  2) HCP ASL (Human Connectome Project, requires free registration + DUA):"
echo "       https://db.humanconnectome.org/"
echo
echo "  3) ISMRM 2015 ASL challenge data (community reconstruction challenge)."
echo
echo "  4) UK Biobank ASL pilot data (requires an approved UK Biobank application)."
echo
echo "For (2)-(4) this script intentionally does NOT attempt to bypass login/DUA."
echo "Register through the portal, then convert one subject's multi-delay delta-M"
echo "series into the loader format documented in data/README.md."
echo
echo "Bigger SYNTHETIC study (no download): "
echo "  python scripts/make_synthetic.py --voxels 1000000"
