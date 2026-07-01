#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.2 -- Whole-Heart Digital Twin   (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size, and NEVER bypasses credentials/registration. This project's demo needs
# NO external data -- its input is a tiny synthetic ensemble config
# (scripts/make_synthetic.py). The datasets below are the REAL-WORLD sources a
# full patient-specific twin is built from; most require registration.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.2 -- Whole-Heart Digital Twin"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project needs NO download: the demo runs on the tiny"
echo "synthetic ensemble config in data/sample/heart_ensemble.txt."
echo "Regenerate or resize it with:"
echo "    python scripts/make_synthetic.py --n 256"
echo
echo "REAL-WORLD datasets a full cardiac digital twin is built from"
echo "(geometry + fibers + calibration targets); most need registration:"
echo "  * UK Biobank Cardiac MRI (100k+ cine CMR) -- https://www.ukbiobank.ac.uk  [application required]"
echo "  * Zenodo Synthetic Biventricular Heart Meshes (1000 meshes) -- https://zenodo.org/records/4506930  [open, CC-BY]"
echo "  * Visible Human Project (CT/MRI/cryosection) -- https://www.nlm.nih.gov/research/visible/visible_human.html  [license/registration]"
echo "  * ACDC MICCAI (100-patient CMR segmentations) -- https://www.creatis.insa-lyon.fr/Challenge/acdc/  [registration]"
echo
echo "None are fetched automatically: credentialed sets must be obtained by"
echo "the user under their own agreement (CLAUDE.md section 8)."
