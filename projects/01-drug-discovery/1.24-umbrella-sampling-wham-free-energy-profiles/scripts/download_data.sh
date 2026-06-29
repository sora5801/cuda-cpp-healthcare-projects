#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real umbrella-sampling pointers (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 1.24 : Umbrella Sampling / WHAM Free Energy Profiles
#
# There is NOTHING to download for this project: the "data" is the synthetic
# experiment configuration in data/sample/umbrella.txt, which the program turns
# into biased trajectories on the fly. This script just points at the real-world
# datasets/tools and never bypasses any registration (CLAUDE.md §8).
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 1.24 -- Umbrella Sampling / WHAM Free Energy Profiles"
echo
echo "There is no file to download: the program derives every window's"
echo "biased trajectory from data/sample/umbrella.txt (a synthetic double-well)."
echo
echo "For REAL umbrella sampling (all-atom MD per window, then WHAM):"
echo "  GROMACS tutorial : https://tutorials.gromacs.org        (gmx wham worked example)"
echo "  SAMPL challenges : https://github.com/samplchallenges/SAMPL  (binding free energy)"
echo "  BindingDB        : https://www.bindingdb.org           (measured affinities)"
echo "  PLUMED           : https://github.com/plumed/plumed2    (collective variables + restraints)"
echo
echo "Bigger SYNTHETIC experiment (no download):"
echo "  python scripts/make_synthetic.py --n-windows 51 --n-sample 200000"
echo
echo "Target data dir: $PROJECT_ROOT/data"
