#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.27 : Parameter Estimation & Data Assimilation for Physiological Models
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The clinical waveform datasets below all
# require registration/credentialed access, so this script only prints
# instructions + links and defers to scripts/make_synthetic.py for the offline,
# fully-reproducible synthetic stand-in the demo actually uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.27 -- Parameter Estimation & Data Assimilation"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The demo runs entirely on the committed SYNTHETIC sample (data/sample/enkf_config.txt)."
echo "Real clinical waveform / cardiac-parameter datasets (all require registration):"
echo "  * PhysioNet MIMIC clinical waveforms  https://physionet.org  (credentialed)"
echo "  * UK Biobank cardiac functional params https://www.ukbiobank.ac.uk  (application)"
echo "  * Zenodo cardiac mechanics emulation   https://zenodo.org/records/7075055"
echo "  * openCARP community experiments       https://opencarp.org/community/community-experiments"
echo
echo "This script does NOT attempt to bypass any credential wall (CLAUDE.md 8)."
echo "For a larger SYNTHETIC problem instead, run:"
echo "    python scripts/make_synthetic.py --ensemble 1024 --windows 80"
