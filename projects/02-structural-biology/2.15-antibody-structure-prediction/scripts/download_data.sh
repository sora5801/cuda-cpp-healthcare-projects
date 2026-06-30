#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.15 : Antibody Structure Prediction  (reduced-scope: CDR screening)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The real antibody databases each have their
# own access terms, so this script PRINTS INSTRUCTIONS ONLY and defers to
# scripts/make_synthetic.py for the offline stand-in the demo actually uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.15 -- Antibody Structure Prediction (CDR screening)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project ships a SYNTHETIC sample (data/sample/antibodies_sample.txt)"
echo "so the demo runs offline. To work with REAL antibody CDR sequences/structures,"
echo "obtain them from the sources below and convert to the loader's text format"
echo "(see data/README.md). Each source has its own license -- respect it."
echo
echo "  SAbDab (Structural Antibody Database) -- IMGT-numbered antibody structures:"
echo "    https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/"
echo "  Thera-SAbDab (therapeutic antibodies):"
echo "    https://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/"
echo "  OAS (Observed Antibody Space) -- ~2 billion antibody sequences:"
echo "    https://opig.stats.ox.ac.uk/webapps/oas/"
echo
echo "  To convert a real set to the screen's format you would:"
echo "    1) IMGT-number each Fv (e.g. with ANARCI) to delimit the six CDR loops,"
echo "    2) emit one line per antibody: '<name> H1 H2 H3 L1 L2 L3' (amino-acid strings),"
echo "    3) put one antibody on a 'QUERY <name> ...' line to screen the rest against."
echo
echo "  For a larger SYNTHETIC problem right now (no download needed):"
echo "    python scripts/make_synthetic.py --n 1048576"
