#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.7 : Monte Carlo Protein Structure Sampling (HP lattice model)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This reduced-scope teaching model
# folds a SYNTHETIC HP sequence, so no external download is required to run the
# demo. The links below point to real-world benchmarks for going further.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.7 -- Monte Carlo Protein Structure Sampling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This reduced-scope HP-lattice demo needs NO download: data/sample/hp_problem.txt"
echo "is a tiny SYNTHETIC HP sequence and the program folds it offline."
echo
echo "For a different SYNTHETIC problem (longer chain, more replicas), run:"
echo "    python scripts/make_synthetic.py --sequence HHPPHPPHPPHPPHPPHPPHPPHH --replicas 1024"
echo
echo "Real-world folding/sampling benchmarks (study these to go further):"
echo "  * CASP structure-prediction benchmarks : https://predictioncenter.org"
echo "  * PDB experimental structures          : https://www.rcsb.org"
echo "  * Dunbrack backbone-dependent rotamers : https://dunbrack.fccc.edu/bbdep2010/"
echo "  * CAMEO continuous benchmarking        : https://www.cameo3d.org"
echo
echo "These are full 3-D coordinate / rotamer datasets that a production MC engine"
echo "(Rosetta, OpenMM) consumes; this 2-D HP teaching model does not parse them."
echo "Respect each site's license/registration -- this script never bypasses it."
