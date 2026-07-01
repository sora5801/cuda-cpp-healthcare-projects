#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.16 : Functional MRI Analysis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. Real fMRI is
# credentialed and/or huge, so this project ships a SYNTHETIC sample and this
# script only prints where to get real data + defers to make_synthetic.py.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.16 : Functional MRI Analysis"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project runs on a SYNTHETIC sample (data/sample/fmri_sample.txt),"
echo "so no download is required for the demo. Real public fMRI sources:"
echo "  * HCP        https://db.humanconnectome.org/   (registration required)"
echo "  * OpenNeuro  https://openneuro.org/            (BIDS; many open datasets)"
echo "  * ABIDE      http://fcon_1000.projects.nitrc.org/indi/abide/"
echo "  * UK Biobank https://www.ukbiobank.ac.uk/      (application + approval)"
echo
echo "Respect every dataset license; credentialed sets are NOT redistributed here."
echo
echo "For a larger SYNTHETIC problem (no download, fully reproducible):"
echo "    python scripts/make_synthetic.py --V 200 --T 240"
echo
echo "To wire a REAL dataset later, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
