#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.18 : ECG Forward Problem & Body-Surface Potential Mapping
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URL +
# expected size, and NEVER bypasses credentials/registration. The real datasets
# here are registration-gated or ship large 3-D torso meshes, so this script
# only prints guidance and defers to scripts/make_synthetic.py for the offline,
# clearly-synthetic stand-in the demo actually uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.18 -- ECG Forward Problem & Body-Surface Potential Mapping"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project runs on a tiny SYNTHETIC sample (data/sample/ecg_sample.txt),"
echo "so no download is required to build, run, and verify the demo."
echo
echo "Real-world data sources (study these; most need registration or are large):"
echo "  * PhysioNet ECG databases          https://physionet.org"
echo "      -- recorded surface ECGs (credentialed for some sets)."
echo "  * EDGAR body-surface potential DB   https://edgar.sci.utah.edu  (verify URL)"
echo "      -- multi-lead body-surface potential maps + torso geometries."
echo "  * Visible Human torso geometry      https://www.nlm.nih.gov/research/visible/visible_human.html"
echo "      -- a realistic torso volume conductor mesh (license/registration)."
echo "  * Cardioid (LLNL) ECG module        https://github.com/llnl/cardioid"
echo "  * openCARP ECG lead calculation     https://git.opencarp.org/openCARP/openCARP"
echo
echo "For a larger SYNTHETIC problem (more electrodes/sources/frames), run:"
echo "  python scripts/make_synthetic.py --L 64 --S 8 --T 500"
echo
echo "When wiring a real dataset later, keep this script idempotent:"
echo "  1) skip download if the file already exists with the right checksum"
echo "  2) print source URL + expected size + SHA256"
echo "  3) for credentialed sets, print registration instructions ONLY (never bypass)"
