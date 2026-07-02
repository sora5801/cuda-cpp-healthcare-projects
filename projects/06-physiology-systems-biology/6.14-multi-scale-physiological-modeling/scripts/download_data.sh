#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.14 : Multi-Scale Physiological Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This project is SIMULATION-ONLY: the demo
# needs no download (the tiny synthetic sample in data/sample/ is enough). This
# script therefore just points you at the real multi-scale model repositories a
# production VPH workflow would draw cell/tissue models from.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.14 -- Multi-Scale Physiological Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project is simulation-only; no dataset download is required to run"
echo "the demo. The committed synthetic sample (data/sample/cable.txt) suffices."
echo
echo "Real multi-scale physiology model repositories (study these):"
echo "  * Physiome Model Repository (CellML cell models):"
echo "      https://models.physiomeproject.org"
echo "  * BioModels Database (systems-biology / ODE models):"
echo "      https://www.ebi.ac.uk/biomodels"
echo "  * OpenCMISS examples (multi-scale FEM setups):"
echo "      https://github.com/OpenCMISS/examples"
echo "  * UK Biobank multi-modal phenotyping (CREDENTIALED -- do NOT bypass;"
echo "    apply for access at):"
echo "      https://www.ukbiobank.ac.uk"
echo
echo "For a larger SYNTHETIC cable (more nodes / longer run), use:"
echo "  python scripts/make_synthetic.py --n 512 --steps 20000"
