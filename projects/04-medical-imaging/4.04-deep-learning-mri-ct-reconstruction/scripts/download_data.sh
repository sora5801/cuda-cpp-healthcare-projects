#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.4 : Deep-Learning MRI/CT Reconstruction  (REDUCED-SCOPE TEACHING VERSION)
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URL +
# expected size, and NEVER bypasses credentials/registration. fastMRI requires a
# signed data-use agreement, so this script prints instructions + links only and
# defers to scripts/make_synthetic.py for an offline synthetic stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.4 : Deep-Learning MRI/CT Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This demo ships a tiny SYNTHETIC acquisition (data/sample/mri_scan_sample.txt);"
echo "no download is required to build or run it."
echo
echo "To study REAL learned reconstruction, get raw multi-coil k-space from fastMRI:"
echo "  1) Register + accept the data-use agreement at:  https://fastmri.med.nyu.edu/"
echo "     (NYU Langone; free for research. We do NOT and CANNOT bypass this step.)"
echo "  2) You receive time-limited download links by email. The knee/brain single- and"
echo "     multi-coil sets are large (tens to hundreds of GB) and are .h5 (HDF5) files."
echo "  3) fastMRI+ radiologist annotations: https://github.com/StanfordMIMI/fastMRI_plus"
echo "  4) For learned CT instead, see the 2016 AAPM Low-Dose CT Grand Challenge."
echo
echo "Reading .h5 k-space + training an E2E-VarNet is a PyTorch task; this C++ demo"
echo "teaches the unrolled-reconstruction STRUCTURE on a small synthetic phantom instead."
echo
echo "Regenerate / resize the synthetic sample with:"
echo "    python scripts/make_synthetic.py --ny 32 --nx 32"
