# ===========================================================================
# scripts/download_data.ps1  --  Real T1D data + simulator pointers (Windows)
# ---------------------------------------------------------------------------
# Project 6.23 : Glucose-Insulin Dynamics & Artificial Pancreas
#
# There is NOTHING to download for the demo: the input is a cohort CONFIG that
# the program expands into virtual patients (data/sample/cohort_params.txt).
# This script prints where the real CGM/insulin datasets and reference simulators
# live. Per CLAUDE.md §8 it never bypasses credentials -- the clinical datasets
# below require registration / a data-use agreement, so we only link them.
#
# Usage:  ./scripts/download_data.ps1
# ===========================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[download_data] Project 6.23 -- Glucose-Insulin Dynamics & Artificial Pancreas"
Write-Host ""
Write-Host "No file to download: the program derives every virtual patient from the"
Write-Host "sweep in data/sample/cohort_params.txt (regenerate with make_synthetic.py)."
Write-Host ""
Write-Host "REAL clinical datasets (require registration / data-use agreement --"
Write-Host "this script will NOT bypass that; apply at the links):"
Write-Host "  OhioT1DM   : https://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html"
Write-Host "               (12-week CGM + insulin for 12 T1D subjects)"
Write-Host "  JAEB CGMS  : https://public.jaeb.org"
Write-Host "  DirecNet   : https://public.jaeb.org/direcnet"
Write-Host ""
Write-Host "Reference SIMULATORS (study the FDA-accepted UVA/Padova model):"
Write-Host "  simglucose : https://github.com/jxx123/simglucose   (Python, gym env)"
Write-Host "  GluCoEnv   : https://github.com/chirathyh/GluCoEnv   (GPU RL env)"
Write-Host "  G2P2C      : https://github.com/RL4H/G2P2C           (RL artificial pancreas)"
Write-Host "  OpenAPS    : https://github.com/openaps/oref0        (reference algorithm)"
Write-Host ""
Write-Host "Bigger SYNTHETIC cohort (no download):"
Write-Host "  python scripts/make_synthetic.py --nSI 64 --nSG 64"
Write-Host ""
Write-Host "Target data dir: $ProjectRoot\data"
