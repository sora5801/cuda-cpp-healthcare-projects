#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to the FULL datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.3 : Cryo-EM Single-Particle Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials / data-use agreements. Real cryo-EM particle stacks
# (EMPIAR) are large and some require accepting a license; this script only
# prints instructions + links and defers to scripts/make_synthetic.py for an
# offline stand-in. The committed sample already runs the demo with no download.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.3 -- Cryo-EM Single-Particle Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project runs on a SYNTHETIC 2-D sample (data/sample/);"
echo "no download is required. To explore REAL cryo-EM data:"
echo
echo "  EMDB     3-D density maps (MRC/.map)   https://www.ebi.ac.uk/emdb/"
echo "  EMPIAR   raw particle image stacks     https://www.ebi.ac.uk/empiar/"
echo "  RCSB     atomic models fit into maps   https://www.rcsb.org"
echo "  cryoDRGN benchmark datasets            https://github.com/ml-struct-bio/cryodrgn"
echo
echo "NOTE: EMPIAR entries are tens of GB and some require accepting a"
echo "      data-use agreement. This script does NOT bypass that -- follow"
echo "      the entry's instructions on the EMPIAR website to download."
echo
echo "For a larger SYNTHETIC problem (any size, fully offline), run:"
echo "    python scripts/make_synthetic.py --n 100000"
