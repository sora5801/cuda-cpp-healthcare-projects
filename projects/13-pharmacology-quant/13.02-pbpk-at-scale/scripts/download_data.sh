#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Realistic PBPK pointers (Linux/macOS)
# Project 13.02 : PBPK at Scale. Nothing to download.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 13.02 -- PBPK at Scale"
echo
echo "There is no file to download: the population is sampled from the"
echo "parameters in data/sample/pbpk_params.txt."
echo
echo "For REAL whole-body PBPK (~15 compartments, literature physiology):"
echo "  PK-Sim : https://github.com/Open-Systems-Pharmacology/PK-Sim"
echo "  nvQSP  : https://github.com/NVIDIA-Digital-Bio/nvQSP   (GPU ODE solvers)"
echo "  Open Systems Pharmacology suite: tissue volumes / blood flows databases."
echo
echo "Bigger population (no download):"
echo "  python scripts/make_synthetic.py --patients 100000"
echo
echo "Target data dir: $PROJECT_ROOT/data"
