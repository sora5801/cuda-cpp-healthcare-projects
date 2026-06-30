#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Reference-data pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.27 : Polarizable Water Model GPU Dynamics
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL, and
# NEVER bypasses credentials/registration. This teaching demo needs no download
# (its committed synthetic cluster is complete); this script PRINTS where the
# real data lives and defers to scripts/make_synthetic.py for larger inputs.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.27 -- Polarizable Water Model GPU Dynamics"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This demo needs NO download: data/sample/water_cluster.txt is a complete,"
echo "self-contained SYNTHETIC cluster. The real-world reference data is:"
echo "  * NIST water thermophysical properties (density/dielectric vs T,P):"
echo "      https://webbook.nist.gov/chemistry/fluid/"
echo "  * TIP4P-2005 / SPC/E reference simulation data and force-field params."
echo "  * MB-pol / AMOEBA polarizable parameters & code:"
echo "      MBX        https://github.com/paesanilab/MBX"
echo "      OpenMM     https://github.com/openmm/openmm"
echo "      Tinker-HP  https://github.com/TinkerTools/tinker-hp"
echo
echo "For a larger SYNTHETIC cluster (e.g. 64 waters), run:"
echo "    python scripts/make_synthetic.py --waters 64"
