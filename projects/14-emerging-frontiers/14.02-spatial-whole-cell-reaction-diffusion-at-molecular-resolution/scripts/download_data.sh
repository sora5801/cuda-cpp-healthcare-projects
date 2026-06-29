#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Molecular-resolution RD pointers (Linux/macOS)
# Project 14.02 : Spatial / Whole-Cell Reaction-Diffusion. Nothing to download.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 14.02 -- Spatial / Whole-Cell Reaction-Diffusion"
echo
echo "There is no file to download: the grid is built from the parameters in"
echo "data/sample/grayscott_params.txt."
echo
echo "This flagship is the continuum (grid stencil) TEACHING version. The full"
echo "project is PARTICLE-based reaction-diffusion at molecular resolution:"
echo "  ReaDDy  : https://github.com/readdy/readdy   (GPU particle RD)"
echo "  Smoldyn : https://github.com/ssandrews/Smoldyn"
echo "  MCell   : https://mcell.org/"
echo "  STEPS   : https://github.com/CNS-OIST/STEPS"
echo
echo "Bigger grid (no download):"
echo "  python scripts/make_synthetic.py --nx 256 --ny 256 --steps 12000"
echo
echo "Target data dir: $PROJECT_ROOT/data"
