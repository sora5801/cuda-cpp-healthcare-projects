#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real-structure PBE input pointers (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 2.9 : Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
#
# CONTRACT (CLAUDE.md sec.8): idempotent, documented, prints sources, and NEVER
# bypasses credentials. There is NOTHING to download for the demo: the committed
# data/sample/molecule.pqr (synthetic) is enough. This script tells you how to
# get a REAL protein into the same .pqr-style input format.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.9 -- Poisson-Boltzmann Electrostatics"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Nothing to download: the demo runs on the committed SYNTHETIC sample"
echo "  data/sample/molecule.pqr  (a tiny dipolar 'molecule')."
echo
echo "To run on a REAL protein, produce a .pqr (atoms with partial charge +"
echo "radius) and convert it to this project's 1-line-header format:"
echo "  1) Fetch a structure from the RCSB PDB:    https://www.rcsb.org"
echo "  2) Add charges + radii with PDB2PQR:       https://github.com/Electrostatics/pdb2pqr"
echo "       pdb2pqr30 --ff=AMBER 1abc.pdb 1abc.pqr"
echo "  3) Reformat the ATOM lines (columns x y z q radius) into our file:"
echo "       header: 'natoms n h eps_in eps_out kappa2 iters' then one"
echo "       'x y z q radius' line per atom  (see data/README.md)."
echo
echo "Reference solvers / benchmarks for comparison:"
echo "  APBS    : https://github.com/Electrostatics/apbs   (PB solver + tests)"
echo "  DelPhi  : http://compbio.clemson.edu/delphi"
echo "  OpenMM  : https://github.com/openmm/openmm         (GPU Generalized Born)"
echo
echo "Bigger synthetic problem (no download):"
echo "  python scripts/make_synthetic.py --n 64 --iters 800"
