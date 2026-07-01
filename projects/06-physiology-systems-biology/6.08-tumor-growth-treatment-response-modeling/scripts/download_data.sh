#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real-data calibration pointers (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 6.8 : Tumor Growth & Treatment-Response Modeling
#
# There is NOTHING to download to run this project: the simulation is built
# deterministically from data/sample/tumor_params.txt. This script only prints
# where REAL data would come from to calibrate a model, and never bypasses any
# registration or credentials (CLAUDE.md §8).
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.8 -- Tumor Growth & Treatment-Response Modeling"
echo
echo "There is no file to download: the tumor field is built from the"
echo "parameters in data/sample/tumor_params.txt (see data/README.md)."
echo
echo "This is a TEACHING model. Real mathematical-oncology models calibrate"
echo "the parameters (D, rho, alpha, beta) against imaging + omics:"
echo "  TCGA (multi-omics + imaging) : https://portal.gdc.cancer.gov"
echo "  TCIA (tumor imaging)         : https://www.cancerimagingarchive.net"
echo "  PhysioNet (oncology series)  : https://physionet.org"
echo "  Zenodo (sim datasets)        : search 'tumor growth simulation'"
echo
echo "Some of these require registration; obtain access through their own"
echo "portals -- this script will not attempt to bypass any credentials."
echo
echo "Bigger / different SYNTHETIC runs (no download):"
echo "  python scripts/make_synthetic.py --nx 256 --ny 256 --steps 800"
echo "  python scripts/make_synthetic.py --dose 3 --n-fractions 10"
echo
echo "Target data dir: $DATA_DIR"
