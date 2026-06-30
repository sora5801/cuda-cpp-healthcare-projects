#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real cryo-EM data pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.12 : Flexible Fitting / MDFF
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Nothing to download for the demo -- the
# committed sample is a self-contained synthetic problem. This only points at the
# real public maps/structures a learner would fit for real.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.12 -- Flexible Fitting / MDFF"
echo
echo "There is no file to download: the demo runs on a self-contained synthetic"
echo "problem in data/sample/mdff_problem.txt (atoms + parameters; the density"
echo "grid is rebuilt by the program)."
echo
echo "For REAL cryo-EM flexible fitting you need a density MAP + a starting MODEL:"
echo "  EMDB   : https://www.ebi.ac.uk/emdb/    (reference density maps, MRC/CCP4)"
echo "  EMPIAR : https://www.ebi.ac.uk/empiar/  (raw particle data)"
echo "  PDB    : ribosome MDFF benchmarks 3J7Y, 4V6X  (https://www.rcsb.org)"
echo "  Tools  : NAMD/VMD MDFF (https://www.ks.uiuc.edu/Research/namd/),"
echo "           phenix.real_space_refine, Coot."
echo
echo "Wiring a real map would add an MRC/CCP4 reader (-> rho) and a PDB reader"
echo "(-> x0); the fitting kernel itself is unchanged."
echo
echo "Larger SYNTHETIC problem (no download):"
echo "  python scripts/make_synthetic.py --iters 400"
echo
echo "Target data dir: $DATA_DIR"
