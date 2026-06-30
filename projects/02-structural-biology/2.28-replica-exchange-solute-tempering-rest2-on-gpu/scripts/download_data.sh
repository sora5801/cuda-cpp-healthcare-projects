#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to the FULL datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.28 : Replica Exchange Solute Tempering (REST2) on GPU
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. This teaching project runs entirely
# on the committed SYNTHETIC sample (data/sample/rest2_config.txt); the datasets
# below are where a *real* REST2 study gets its validation data, so this script
# only prints instructions + links and defers to scripts/make_synthetic.py.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.28 -- Replica Exchange Solute Tempering (REST2) on GPU"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project's demo needs NO download: it runs on the committed synthetic"
echo "sample data/sample/rest2_config.txt. Regenerate or sweep it with:"
echo "    python scripts/make_synthetic.py --barrier-h 9 --n-replicas 16"
echo
echo "Real-world REST2 VALIDATION datasets (open the links; respect each license):"
echo "  * Shaw millisecond folding trajectories -- by request/collaboration; not redistributable."
echo "  * SAMPL challenges      : https://github.com/samplchallenges/SAMPL  (open)"
echo "  * GPCRmd REST2 data     : https://gpcrmd.org                        (web access; site terms)"
echo "  * Chignolin / Trp-cage fast-folder benchmarks -- public sequences; standard REMD test systems."
echo
echo "None of these is required for the demo. For credentialed sets, register at the"
echo "source FIRST; this script will never bypass authentication (CLAUDE.md section 8)."
