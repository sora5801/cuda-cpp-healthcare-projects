#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch/point-to the FULL datasets (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 4.17 : Real-Time Intraoperative / Image-Guided Surgery
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. The real IGS datasets below are
# video/volume corpora behind registration or challenge sign-ups, so this
# script only prints instructions + links. The committed synthetic sample in
# data/sample/ is all the demo needs; make_synthetic.py can scale it up.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.17 -- Real-Time Intraoperative / Image-Guided Surgery"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project runs on a SYNTHETIC point-cloud pair (data/sample/surface_pair.txt)."
echo "No download is required to build, run, or verify the demo."
echo
echo "For a LARGER synthetic problem (deterministic), run:"
echo "    python scripts/make_synthetic.py --grid 40 --noise 0.3"
echo
echo "Real image-guided-surgery datasets (registration / credentials required --"
echo "this script does NOT bypass any login; it only points you to the source):"
echo "  * Cholec80 laparoscopic videos : https://camma.u-strasbg.fr/datasets"
echo "  * ReMIND2Reg 2025 (brain)      : https://arxiv.org/abs/2508.09649"
echo "  * EndoVis (MICCAI) challenges  : https://endovis.grand-challenge.org/"
echo "  * SurgT tool-tracking benchmark"
echo
echo "To use a real surface: sample 3-D points from the two surfaces and write"
echo "them in data/sample/surface_pair.txt's format (see data/README.md)."
