#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Point at the real datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.34 : Amyloid / Aggregation Propensity Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The demo runs entirely on the tiny
# SYNTHETIC FASTA committed in data/sample/, so no download is required to learn
# from this project. This script tells you where the real curated aggregation
# datasets live and how to use them with the binary.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.34 -- Amyloid / Aggregation Propensity Prediction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed tiny SYNTHETIC sample (data/sample/amyloid_sample.fasta)"
echo "is enough to run the demo offline. The binary accepts any FASTA file:"
echo "    ./amyloid-aggregation-propensity-prediction <your.fasta>"
echo
echo "Real curated aggregation datasets (study these; respect each license):"
echo "  * AmyPro       -- curated amyloidogenic protein regions (FASTA + annotations)"
echo "                    https://amypro.net   (downloadable; cite Varadi et al. 2018)"
echo "  * WALTZ-DB 2.0 -- experimental hexapeptide amyloid/non-amyloid labels"
echo "                    https://waltzdb.switchlab.org"
echo "  * EMDB fibril cryo-EM maps (structural validation of predicted APRs)"
echo "                    https://www.ebi.ac.uk/emdb/"
echo
echo "How to use a real set with this teaching tool:"
echo "  1) Download the sequences as a plain FASTA file (one '>' header per protein)."
echo "  2) Run the binary on it; it scans every sequence and ranks them by APR."
echo "  3) Compare the predicted hot spots against the database's annotated"
echo "     amyloidogenic regions -- that is the natural next exercise (README)."
echo
echo "Idempotent download pattern to follow if you script a real fetch:"
echo "  - skip if the file already exists with the expected SHA256;"
echo "  - print source URL + expected size + checksum before downloading;"
echo "  - for any credentialed source, print registration instructions ONLY."
