#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 1.5 : Free Energy Perturbation / Thermodynamic Integration
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This project is a REDUCED-SCOPE teaching
# model with no real dataset to fetch -- a faithful FEP/TI run needs a full MD
# engine + force field. So this script only prints links and defers to the
# committed synthetic sample (scripts/make_synthetic.py) for an offline run.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.5 -- Free Energy Perturbation / Thermodynamic Integration"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This is a REDUCED-SCOPE teaching model: it samples a 1-D harmonic"
echo "alchemical system whose DeltaG has a CLOSED FORM, so no external data"
echo "is required. The committed synthetic sample runs the demo offline."
echo
echo "Real FEP/TI benchmarks to study (each needs a full MD engine):"
echo "  * Merck FEP benchmark set (open, via OpenFE):"
echo "      https://github.com/OpenFreeEnergy/openfe"
echo "  * FEP+ validation set (Schrodinger; registration required) -- links only."
echo "  * PDBbind experimental binding affinities:  http://www.pdbbind.org.cn"
echo "  * ChEMBL bioactivity data:                  https://www.ebi.ac.uk/chembl/"
echo
echo "For a different SYNTHETIC problem, run:"
echo "    python scripts/make_synthetic.py --kB 9 --windows 21"
echo
echo "NOTE: credentialed sets (FEP+, some PDBbind tiers) require you to register"
echo "and accept their license yourself; this script will never bypass that."
