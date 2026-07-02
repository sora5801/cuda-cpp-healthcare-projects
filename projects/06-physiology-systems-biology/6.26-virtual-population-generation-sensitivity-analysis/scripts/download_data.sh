#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Realistic virtual-population pointers (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 6.26 : Virtual Population Generation & Sensitivity Analysis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. There is nothing to download for the demo -- the
# study is generated from data/sample/vpop_config.txt. This script only prints
# where the REAL physiology/PBPK data lives and defers to make_synthetic.py for
# a larger offline study.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.26 -- Virtual Population Generation & Sensitivity Analysis"
echo
echo "There is no file to download: the demo's virtual population is"
echo "generated deterministically from data/sample/vpop_config.txt."
echo
echo "For a REALISTIC virtual population + sensitivity workflow, use:"
echo "  NHANES physiology  : https://www.cdc.gov/nchs/nhanes/"
echo "  WHO growth data    : https://www.who.int/tools/growth-reference-data-for-5to19-years"
echo "  OSP PBPK library   : https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library"
echo "  FDA drug-label PK  : https://www.fda.gov/drugs"
echo "  SALib (reference)  : https://github.com/SALib/SALib"
echo
echo "These are externally licensed; respect each source's terms. This"
echo "script does NOT attempt to bypass any registration."
echo
echo "Bigger SYNTHETIC study (no download):"
echo "  python scripts/make_synthetic.py --N 16384"
echo
echo "Target data dir: $DATA_DIR"
