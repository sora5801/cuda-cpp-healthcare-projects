#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.8 -- Semi-Empirical & Tight-Binding Quantum Methods
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size, and NEVER bypasses credentials/registration. The real datasets are large
# molecular sets with 3-D geometries; converting them to pi-system graphs needs a
# chemistry toolkit (RDKit / Open Babel) and is out of scope here. This script
# prints guidance and links only; the demo runs on the committed synthetic sample
# (or scripts/make_synthetic.py).
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.8 -- Semi-Empirical & Tight-Binding Quantum Methods"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project runs on a tiny SYNTHETIC sample (data/sample/molecules_sample.txt)"
echo "and needs NO download. The real datasets the method is benchmarked on are:"
echo
echo "  ANI-1   ~20M DFT energies on 57k molecules   https://github.com/isayev/ANI1   (CC0)"
echo "  QM9     134k molecules + DFT properties       https://doi.org/10.6084/m9.figshare.978904"
echo "  GMTKN55 thermochemistry/kinetics benchmark    https://www.chemie.uni-bonn.de/grimme/de/software/gmtkn"
echo "  COMPAS  polycyclic aromatic systems           (verify current URL)"
echo
echo "These ship 3-D geometries; convert each molecule to a pi-system graph with a chemistry"
echo "toolkit (RDKit/Open Babel) to feed this project's loader. None require bypassing any"
echo "credentials; respect each dataset's license."
echo
echo "To (re)generate the committed synthetic batch instead, run:"
echo "    python scripts/make_synthetic.py"
