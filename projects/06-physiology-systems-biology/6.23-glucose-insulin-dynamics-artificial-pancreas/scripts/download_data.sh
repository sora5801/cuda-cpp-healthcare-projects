#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real T1D data + simulator pointers (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 6.23 : Glucose-Insulin Dynamics & Artificial Pancreas
#
# There is NOTHING to download for the demo: the input is a cohort CONFIG that
# the program expands into virtual patients (data/sample/cohort_params.txt).
# This script prints where the real CGM/insulin datasets and reference simulators
# live. Per CLAUDE.md §8 it never bypasses credentials -- the clinical datasets
# below require registration / a data-use agreement, so we only link them.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 6.23 -- Glucose-Insulin Dynamics & Artificial Pancreas"
echo
echo "No file to download: the program derives every virtual patient from the"
echo "sweep in data/sample/cohort_params.txt (regenerate with make_synthetic.py)."
echo
echo "REAL clinical datasets (require registration / data-use agreement --"
echo "this script will NOT bypass that; apply at the links):"
echo "  OhioT1DM   : https://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html"
echo "               (12-week CGM + insulin for 12 T1D subjects)"
echo "  JAEB CGMS  : https://public.jaeb.org"
echo "  DirecNet   : https://public.jaeb.org/direcnet"
echo
echo "Reference SIMULATORS (study the FDA-accepted UVA/Padova model):"
echo "  simglucose : https://github.com/jxx123/simglucose   (Python, gym env)"
echo "  GluCoEnv   : https://github.com/chirathyh/GluCoEnv   (GPU RL env)"
echo "  G2P2C      : https://github.com/RL4H/G2P2C           (RL artificial pancreas)"
echo "  OpenAPS    : https://github.com/openaps/oref0        (reference algorithm)"
echo
echo "Bigger SYNTHETIC cohort (no download):"
echo "  python scripts/make_synthetic.py --nSI 64 --nSG 64"
echo
echo "Target data dir: $PROJECT_ROOT/data"
