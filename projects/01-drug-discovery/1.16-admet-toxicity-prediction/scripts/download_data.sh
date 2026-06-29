#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.16 : ADMET / Toxicity Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. The public ADMET sets below are
# downloadable directly, but turning molecules into the descriptor format this
# project expects requires a featurizer (RDKit / Chemprop) you run yourself, so
# this script prints the recipe and links rather than fabricating data. The
# committed synthetic sample in data/sample/ is enough to run the demo offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.16 -- ADMET / Toxicity Prediction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Public ADMET / toxicity datasets (fetch yourself; respect each license):"
echo "  * Tox21 (12 endpoints, ~8k compounds): https://tripod.nih.gov/tox21/"
echo "  * TDC ADMET benchmark group:           https://tdcommons.ai/benchmark/admet_group/overview/"
echo "  * ClinTox (FDA approved vs failed):    https://moleculenet.org"
echo "  * DILI (drug-induced liver injury):    search current literature for a redistributable release"
echo
echo "To turn SMILES/molecules into the '<name> <descriptor...>' format this project reads:"
echo "  1) featurize with RDKit descriptors or Chemprop D-MPNN features"
echo "  2) write one line per molecule: '<name> <x_0> ... <x_{D-1}>' (D = ADMET_D in src/admet_core.h)"
echo "  3) prepend the M trained endpoint models as '<endpoint> <bias> <w_0> ... <w_{D-1}>'"
echo
echo "No credentialed download is attempted. For an OFFLINE synthetic stand-in:"
echo "    python scripts/make_synthetic.py --n 1000000"
echo
echo "Idempotency pattern when wiring a real fetch: skip if the file already exists"
echo "with the right SHA256; print source URL + expected size + checksum; for"
echo "credentialed sets print registration instructions ONLY (never bypass them)."
