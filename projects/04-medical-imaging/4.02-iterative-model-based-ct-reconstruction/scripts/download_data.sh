#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.2 : Iterative / Model-Based CT Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + licensing,
# and NEVER bypasses credentials/registration. The real low-dose CT datasets are
# all credentialed / non-redistributable, so this script prints instructions +
# links and defers to scripts/make_synthetic.py for the offline synthetic sample.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.2 -- Iterative / Model-Based CT Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The real datasets for this project are CREDENTIALED and NOT redistributable."
echo "This script does not (and must not) bypass their registration. To obtain them:"
echo
echo "  * 2016 AAPM Low-Dose CT Grand Challenge (paired low/normal-dose scans)"
echo "      https://www.aapm.org/grandchallenge/lowdosect/   (register for access)"
echo "  * Mayo Clinic Low-Dose CT  -- via TCIA (The Cancer Imaging Archive)"
echo "  * LIDC-IDRI CT scans       -- via TCIA, under a data-use agreement"
echo "      https://www.cancerimagingarchive.net/"
echo
echo "After downloading (with your own credentials), convert a scan's sinogram to"
echo "the text format documented in data/README.md."
echo
echo "The committed synthetic sample (data/sample/sinogram_sample.txt) is enough to"
echo "run the demo offline. For a larger SYNTHETIC problem, run:"
echo "    python scripts/make_synthetic.py --angles 90 --det 127 --img 96 --iters 80"
