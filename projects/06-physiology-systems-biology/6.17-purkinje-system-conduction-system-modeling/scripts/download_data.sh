#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.17 -- Purkinje System & Conduction System Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This project ships a reduced-scope
# TEACHING model that runs entirely on the committed synthetic sample, so there
# is no required download. This script points to the real research datasets and
# defers to scripts/make_synthetic.py for the offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.17 -- Purkinje System & Conduction System Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project runs on the committed SYNTHETIC sample (data/sample/purkinje_tree.txt)."
echo "No download is required to build, run, or verify the demo."
echo
echo "To regenerate / enlarge the synthetic tree:"
echo "    python scripts/make_synthetic.py"
echo
echo "Real Purkinje-network geometries and His-bundle electrograms (study these to"
echo "extend the project -- see README 'Exercises'):"
echo "  * openCARP community Purkinje experiments:"
echo "      https://opencarp.org/community/community-experiments"
echo "  * MonoAlg3D_C Purkinje examples (GPU monodomain + PMJ calibration):"
echo "      https://github.com/rsachetto/MonoAlg3D_C"
echo "  * NeuroMorpho (branching-tree morphologies, analogy):"
echo "      https://neuromorpho.org"
echo "  * PhysioNet His-bundle electrogram databases (may require registration):"
echo "      https://physionet.org"
echo
echo "NOTE: PhysioNet and similar sources may require an account + a signed data-use"
echo "agreement. This script prints instructions ONLY and never bypasses that step."
