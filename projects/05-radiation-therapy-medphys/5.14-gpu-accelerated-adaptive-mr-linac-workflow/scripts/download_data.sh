#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 5.14 : GPU-Accelerated Adaptive MR-Linac Workflow (reduced-scope)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real MR-Linac images are patient data and
# cannot be redistributed here, so this script prints where to obtain them and
# defers to scripts/make_synthetic.py for the offline synthetic stand-in the demo
# actually uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.14 -- GPU-Accelerated Adaptive MR-Linac Workflow"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/oart_case.txt) is SYNTHETIC and is all the"
echo "demo needs. Real MR-Linac data is patient-derived and NOT redistributed here."
echo
echo "To work with real MR-guided radiotherapy images, obtain them yourself from:"
echo "  * MR-Linac Consortium shared datasets  -> verify URL at mrlinac.org (access)"
echo "  * TCIA MR-guided RT collections        -> https://www.cancerimagingarchive.net/"
echo "                                            (per-collection license / DUA)"
echo "  * AAPM MR-Linac Working Group test cases -> AAPM task-group pages"
echo "  * MRI-only radiotherapy cohorts        -> per published-paper terms"
echo
echo "Respect every license; some require registration or a data-use agreement."
echo "This script intentionally does NOT bypass any of that."
echo
echo "For a larger SYNTHETIC slice instead, run:"
echo "    python scripts/make_synthetic.py --nx 64 --ny 64"
