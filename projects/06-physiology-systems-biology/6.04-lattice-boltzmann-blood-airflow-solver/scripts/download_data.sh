#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Realistic LBM geometry pointers (Linux/macOS)
# Project 6.04 : Lattice-Boltzmann Blood/Airflow Solver. Nothing to download.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 6.04 -- Lattice-Boltzmann Blood/Airflow Solver"
echo
echo "There is no file to download: the solver makes its own flow from the"
echo "parameters in data/sample/channel_params.txt."
echo
echo "For REAL 3-D geometry (segmented vessels/airways + D3Q19/D3Q27):"
echo "  HemeLB     : https://github.com/hemelb-codes/hemelb"
echo "  PALABOS    : https://gitlab.com/unigespc/palabos"
echo "  USERMESO-2 : https://github.com/AnselGitAccount/USERMESO-2.0"
echo
echo "Bigger 2-D grid:"
echo "  python scripts/make_synthetic.py --nx 128 --ny 64 --steps 20000"
echo
echo "Target data dir: $PROJECT_ROOT/data"
