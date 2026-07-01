#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Realistic PK/PD & PBPK pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.15 : PK/PD & PBPK Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. There is NOTHING to download for this project: the
# virtual population is sampled from the parameters in data/sample/pkpd_params.txt.
# This script only prints where to get REAL PK/PD data and models.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.15 -- PK/PD & PBPK Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "There is no file to download: the virtual population is generated from"
echo "the parameters in data/sample/pkpd_params.txt (see scripts/make_synthetic.py)."
echo
echo "For REAL clinical PK data and validated PK/PD & PBPK models:"
echo "  PhysioNet / MIMIC (clinical time series; CREDENTIALED -- register, do not scrape):"
echo "    https://physionet.org"
echo "  FDA FAERS (adverse-event reports, public):"
echo "    https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers"
echo "  OSP PBPK Model Library (whole-body PBPK models):"
echo "    https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library"
echo "  DDMoRe model repository (curated pharmacometric models):"
echo "    https://ddmore.eu/models-tools"
echo
echo "Bigger SYNTHETIC population (no download):"
echo "  python scripts/make_synthetic.py --patients 100000"
