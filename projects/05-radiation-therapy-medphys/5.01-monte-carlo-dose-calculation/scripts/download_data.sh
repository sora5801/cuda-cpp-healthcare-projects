#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Realistic MC physics pointers (Linux/macOS)
# Project 5.01 : Monte Carlo Dose Calculation (simplified slab). Nothing to fetch.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 5.01 -- Monte Carlo Dose Calculation (simplified slab)"
echo
echo "There is no file to download: the simulation makes its own data from"
echo "the parameters in data/sample/mc_params.txt."
echo
echo "For REAL physics (cross sections, electron transport, CT geometry):"
echo "  EGSnrc : https://github.com/nrc-cnrc/EGSnrc   (reference MC + PEGS data)"
echo "  GATE   : https://github.com/OpenGATE/opengate (Geant4 clinical MC)"
echo "  MC-GPU : https://github.com/DIDSR/MCGPU        (open CUDA photon MC)"
echo
echo "More histories (smoother statistics):"
echo "  python scripts/make_synthetic.py --photons 4000000"
echo
echo "Target data dir: $PROJECT_ROOT/data"
