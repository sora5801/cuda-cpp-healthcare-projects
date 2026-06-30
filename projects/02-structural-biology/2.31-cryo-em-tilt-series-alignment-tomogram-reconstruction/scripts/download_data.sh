#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.31 -- Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size, and NEVER bypasses credentials/registration. This project runs fully on
# the committed SYNTHETIC sample; the real cryo-ET archives below are large, so
# this script only prints where to get them.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.31 -- Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "  This project ships a TINY SYNTHETIC sample (data/sample/tilt_series_sample.txt)"
echo "  that is enough to build and run the demo OFFLINE. No download is required."
echo
echo "  Real cryo-ET tilt series are large (multi-GB) research archives:"
echo "    * EMPIAR tilt-series archives    https://www.ebi.ac.uk/empiar/"
echo "        e.g. EMPIAR-10045 (in-situ ribosome tilt series)"
echo "    * EMDB subtomogram averages      https://www.ebi.ac.uk/emdb/"
echo "    * SHREC cryo-ET benchmark        (verify the current URL on the SHREC site)"
echo "  Respect each entry's license before redistributing. These are NOT fetched"
echo "  here (size + per-entry terms); see data/README.md for how to adapt them to"
echo "  this project's simple text layout."
echo
echo "  For a larger SYNTHETIC problem instead, run:"
echo "    python scripts/make_synthetic.py --maxtilt 70 --step 4 --det 257 --img 192"
