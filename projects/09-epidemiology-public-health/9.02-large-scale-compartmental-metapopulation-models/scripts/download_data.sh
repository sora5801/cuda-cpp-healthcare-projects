#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Realistic epidemic-model pointers (Linux/macOS)
# Project 9.02 : Large-Scale Compartmental & Metapopulation Models. Nothing to fetch.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 9.02 -- Large-Scale Compartmental & Metapopulation Models"
echo
echo "There is no file to download: the program derives every member's"
echo "parameters from the sweep in data/sample/ensemble_params.txt."
echo
echo "For REAL models (mobility matrices, age structure, many patches):"
echo "  MEmilio     : https://github.com/SciCompMod/memilio   (C++/CUDA)"
echo "  EpiModel    : https://github.com/EpiModel/EpiModel    (R, network)"
echo "  Torchdiffeq : https://github.com/rtqichen/torchdiffeq (GPU ODE solvers)"
echo
echo "Bigger ensemble (no download):"
echo "  python scripts/make_synthetic.py --nb 200 --ng 200"
echo
echo "Target data dir: $PROJECT_ROOT/data"
