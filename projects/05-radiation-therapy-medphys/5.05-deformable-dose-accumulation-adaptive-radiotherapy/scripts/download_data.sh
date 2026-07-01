#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch / point to the FULL dataset (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 5.5 : Deformable Dose Accumulation & Adaptive Radiotherapy
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs +
# licensing notes, and NEVER bypasses credentials/registration. The demo does not
# need any of this -- scripts/make_synthetic.py already writes a tiny offline
# sample. This script exists so a learner who wants REAL ART data knows exactly
# where to get it and how to shape it for this project.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.5 -- Deformable Dose Accumulation & Adaptive Radiotherapy"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC offline sample (data/sample/art_case.txt);"
echo "the demo needs nothing downloaded. The datasets below are real-world DIR /"
echo "ART benchmarks -- each requires you to accept a data-use license yourself,"
echo "so this script only PRINTS pointers. It never bypasses any registration."
echo
echo "  * DIR-Lab 4D-CT lung  : https://www.dir-lab.com/"
echo "      Respiratory 4D-CT phase pairs with expert landmarks (the gold-standard"
echo "      target-registration-error benchmark for DIR)."
echo "  * AAPM TG-132 DIR      : https://www.aapm.org/pubs/reports/RPT_132.pdf"
echo "      The clinical QA reference for DIR + deformable dose accumulation."
echo "  * TCIA CT-on-rails/CBCT: https://www.cancerimagingarchive.net/"
echo "      Planning CT + daily CBCT collections for adaptive-radiotherapy studies."
echo "  * CREATIS lung phantom : https://www.creatis.insa-lyon.fr/"
echo "      A deformable lung phantom with a known ground-truth motion field."
echo
echo "To use real data with THIS teaching project, export one 2-D slice pair:"
echo "  planning image, daily image, planning dose, daily dose  (all same nx x ny),"
echo "normalize images to [0,1] and doses to Gy, and write them in the sample's"
echo "text format (see data/README.md). DO NOT commit patient-derived data."
echo
echo "For a larger SYNTHETIC problem instead:"
echo "  python scripts/make_synthetic.py --nx 256 --ny 256 --shift 12.0"
