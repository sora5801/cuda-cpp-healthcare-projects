#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.28 : GPU-Accelerated DRR Generation for 2D/3D Registration
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL and
# NEVER bypasses credentials/registration. The DRR demo runs fully on the
# committed SYNTHETIC phantom (data/sample/ct_volume_sample.txt), so there is no
# mandatory download. This script points at real CT sources and defers to
# make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.28 -- GPU-Accelerated DRR Generation for 2D/3D Registration"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/ct_volume_sample.txt) is SYNTHETIC and is all"
echo "the demo needs. No download is required to build, run, or verify this project."
echo
echo "To experiment on REAL CT volumes, convert a DICOM series into this loader's"
echo "text format (header 'nx ny nz sx sy sz' then nx*ny*nz Hounsfield Units,"
echo "row-major [z][y][x]) using e.g. pydicom or SimpleITK. Public sources:"
echo "  * TCIA (The Cancer Imaging Archive): https://www.cancerimagingarchive.net/"
echo "      prostate/lung CT collections (mostly CC-BY)."
echo "  * Gold Atlas male-pelvis MR/CT (verify URL): https://www.goldenatlasproject.com/"
echo "  * AAPM TG-132 image-registration test cases."
echo "  * Clinical CBCT + kV portal images: institutional IRB only -- NOT redistributed."
echo
echo "For a larger SYNTHETIC volume (no download, no credentials), run:"
echo "    python scripts/make_synthetic.py --n 128      # 128^3 phantom"
echo
echo "This script intentionally downloads nothing automatically: the public sets are"
echo "large and/or require accepting a data-use agreement, which must be done by a"
echo "human (CLAUDE.md section 8). It will never bypass that step."
