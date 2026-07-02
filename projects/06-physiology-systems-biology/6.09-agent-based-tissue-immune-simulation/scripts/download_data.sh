#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  "Fetch the FULL dataset" (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.9 : Agent-Based Tissue / Immune Simulation
#
# This project is a SIMULATION: its input is a scenario parameter file, not a
# measured dataset, so there is nothing to download to run the demo. This script
# (per CLAUDE.md §8) explains where REAL tissue/immune data live for those who
# want to calibrate the model, and NEVER bypasses any registration.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.9 -- Agent-Based Tissue / Immune Simulation"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project generates its own tissue state; NO download is needed to run"
echo "the demo. The committed sample (data/sample/tissue_params.txt) is enough."
echo
echo "For a larger SYNTHETIC scenario, regenerate the parameter file:"
echo "    python scripts/make_synthetic.py --gx 64 --gy 64 --n-tumor 400 --n-immune 300 --steps 800"
echo
echo "To CALIBRATE cell states / immune landscapes against real data, see"
echo "(each requires its own registration / license -- follow their terms):"
echo "  * CancerSEA single-cell functional states : http://biocc.hrbmu.edu.cn/CancerSEA/"
echo "  * TCGA pan-cancer immune landscape        : https://portal.gdc.cancer.gov"
echo "  * MIBI/IMC imaging mass cytometry         : various Zenodo deposits"
echo "  * TCIA immunotherapy imaging              : https://www.cancerimagingarchive.net"
echo
echo "These are NOT auto-downloaded: they are large, credentialed, and their"
echo "licenses forbid blind redistribution. Educational use only -- not clinical."
