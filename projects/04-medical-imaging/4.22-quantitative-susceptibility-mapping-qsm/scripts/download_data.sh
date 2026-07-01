#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch / locate the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.22 : Quantitative Susceptibility Mapping (QSM)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs +
# licensing, and NEVER bypasses credentials/registration. Every real QSM dataset
# below requires registration or carries redistribution limits, so this script
# only PRINTS instructions and links; the committed synthetic sample
# (data/sample/field_map.txt) lets the demo run offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.22 -- Quantitative Susceptibility Mapping (QSM)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The demo runs on the committed SYNTHETIC sample (data/sample/field_map.txt)."
echo "No download is required to build, run, or study this project."
echo
echo "To study REAL QSM data, use one of these sources (each has its own license"
echo "and most require registration -- respect them; we do not redistribute any):"
echo
echo "  * QSM Reconstruction Challenge 2.0 (benchmark data + reference recons):"
echo "      https://doi.org/10.1101/2020.11.25.397695   (data on Zenodo)"
echo "  * HCP 7T multi-echo GRE (Human Connectome Project):"
echo "      https://db.humanconnectome.org/              (registration required)"
echo "  * AHEAD ultra-high-field 7T lifespan database (Amsterdam)."
echo "  * UK Biobank (credentialed):"
echo "      https://www.ukbiobank.ac.uk/                 (application required)"
echo
echo "After obtaining a LOCAL FIELD MAP (phase unwrapped + background removed),"
echo "export the 3-D volume to this project's text format:"
echo "    line 1: 'nx ny nz'"
echo "    then nx*ny*nz field-shift values, x fastest then y then z"
echo "and pass its path to the executable. See data/README.md for details."
echo
echo "For a larger SYNTHETIC field map instead, run:"
echo "    python scripts/make_synthetic.py --nx 24 --ny 24 --nz 16"
