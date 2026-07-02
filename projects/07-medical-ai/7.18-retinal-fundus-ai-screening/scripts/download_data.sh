#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 7.18 : Retinal Fundus AI Screening
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size, and NEVER bypasses credentials/registration. Every real fundus dataset
# here is account-gated, so this prints instructions + links only and defers to
# scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 7.18 -- Retinal Fundus AI Screening"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "All real fundus datasets are ACCESS-RESTRICTED. This script does NOT bypass"
echo "any login or license -- it prints where to get them:"
echo
echo "  EyePACS   ~88,000 labelled fundus images, 5-grade DR severity."
echo "            Kaggle 'Diabetic Retinopathy Detection' (account + rules). Verify URL."
echo "  APTOS2019 3,662 fundus images, DR grading."
echo "            Kaggle 'APTOS 2019 Blindness Detection' (account). Verify URL."
echo "  DRIVE/STARE  retinal vessel-segmentation datasets (registration varies)."
echo "  UK Biobank  ~68k fundus images + linked health records (credentialed):"
echo "            https://www.ukbiobank.ac.uk/"
echo
echo "The committed tiny sample data/sample/fundus_sample.txt runs the demo offline."
echo "To (re)generate a synthetic fundus image instead:"
echo "    python scripts/make_synthetic.py --size 32"
echo
echo "To use a real image: load it (Pillow), resize to a small square, divide by 255,"
echo "and write it in the 'C H W label' + channel-major float format (data/README.md)."
