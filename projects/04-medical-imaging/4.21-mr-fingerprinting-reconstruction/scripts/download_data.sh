#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.21 -- MR Fingerprinting Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs + how to
# obtain each dataset, and NEVER bypasses credentials/registration. The real MRF
# datasets below require registration, so this script only prints instructions +
# links and defers to scripts/make_synthetic.py for an offline stand-in. The
# committed tiny sample already runs the demo with zero downloads.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.21 -- MR Fingerprinting Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a TINY committed SYNTHETIC sample (data/sample/mrf_sample.txt)"
echo "that runs the demo offline. Real MR Fingerprinting datasets require"
echo "registration and are NOT redistributed here. To obtain them yourself:"
echo
echo "  1) fastMRI (includes qMRI/MRF-style data) -- https://fastmri.org/"
echo "     Register and accept the data use agreement; download instructions"
echo "     are provided after approval. (Verify the exact MRF subset URL there.)"
echo
echo "  2) Cleveland Clinic MRF dataset -- search IEEE DataPort (https://ieee-dataport.org/)"
echo "     for 'MR Fingerprinting'; access terms vary per collection (verify URL)."
echo
echo "  3) qMRI.org quantitative-MRI resources -- https://qmri.org/ (verify URL)."
echo
echo "  4) Synthetic phantoms -- generate BrainWeb/XCAT-style ground truth locally."
echo
echo "For a LARGER synthetic problem (more voxels / a bigger dictionary), run:"
echo "    python scripts/make_synthetic.py --V 4096"
echo
echo "When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
echo
echo "[download_data] No files downloaded (by design). The demo runs on the"
echo "[download_data] committed synthetic sample."
