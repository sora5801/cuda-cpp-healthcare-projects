#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.1 -- Cardiac Electrophysiology Simulation   (template skeleton)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. Defers to
# scripts/make_synthetic.py for an offline stand-in when needed.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.1 -- Cardiac Electrophysiology Simulation"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# This project simulates from a parameter file, not a downloaded dataset, so the
# committed synthetic sample is fully self-contained. The catalog datasets below
# are the REAL-WORLD sources you would use to build validated, patient-specific
# cardiac models; none is needed to run this teaching demo.
echo "  This project needs NO download: the committed synthetic sample in"
echo "  data/sample/tissue_params.txt is enough to run the demo end-to-end."
echo
echo "  Real-world data sources (from the catalog) for validated cardiac EP:"
echo "    * PhysioNet MIT-BIH & MIMIC-III Waveform -- ICU ECG/hemodynamics (https://physionet.org)"
echo "    * CellML Physiome Repository -- curated ionic cell models (https://models.physiomeproject.org)"
echo "    * UK Biobank Cardiac MRI -- cine CMR, access via application (https://www.ukbiobank.ac.uk)"
echo "    * ACDC MICCAI Cardiac Challenge -- CMR with myocardium ground truth"
echo "      (https://www.creatis.insa-lyon.fr/Challenge/acdc/)"
echo
echo "  These require registration/credentials; this script never bypasses that."
echo "  For a bigger SYNTHETIC problem, run:"
echo "    python scripts/make_synthetic.py --nx 128 --ny 128 --steps 1200"
echo
echo "  If you later wire a real dataset, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
