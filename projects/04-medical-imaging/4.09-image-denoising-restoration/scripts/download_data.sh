#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.9 : Image Denoising & Restoration  (Non-Local Means)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size, and NEVER bypasses credentials/registration. Every real dataset for this
# project is license-restricted or credentialed, so this script prints
# instructions + links ONLY and defers to scripts/make_synthetic.py for an
# offline stand-in. The committed data/sample/ is enough to run the demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.9 -- Image Denoising & Restoration (Non-Local Means)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed tiny SYNTHETIC sample (data/sample/phantom_sample.txt) is enough"
echo "to build and run the demo offline. The REAL medical datasets below are"
echo "license-restricted or credentialed -- this script only prints how to obtain"
echo "them; it never bypasses any registration (CLAUDE.md section 8)."
echo
echo "  1) 2016 AAPM Low-Dose CT Grand Challenge (quarter/full-dose CT pairs)"
echo "       https://www.aapm.org/grandchallenge/lowdosect/"
echo "       -> agree to the challenge data-use terms, then download the DICOM pairs."
echo "  2) NLST (National Lung Screening Trial) chest CT via TCIA"
echo "       https://www.cancerimagingarchive.net/"
echo "       -> requires a TCIA account + data-use agreement."
echo "  3) Fluorescence Microscopy Noise Dataset (for Noise2Void)"
echo "       https://github.com/juglab/n2v"
echo "  4) SIDD smartphone image-noise dataset (non-medical sanity check)"
echo
echo "For a larger SYNTHETIC problem you can generate right now:"
echo "    python scripts/make_synthetic.py --size 128 --sigma 0.10"
echo
echo "When wiring a real dataset later, keep this idempotent pattern:"
echo "    1) skip the download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
