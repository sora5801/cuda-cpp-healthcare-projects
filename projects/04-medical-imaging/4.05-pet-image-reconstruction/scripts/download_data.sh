#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.5 : PET Image Reconstruction (MLEM / OS-EM)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Real PET reconstruction data is large
# and often gated, so this script prints pointers only and defers to
# scripts/make_synthetic.py for an offline stand-in. The committed tiny sample in
# data/sample/ is already enough to run the demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.5 -- PET Image Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/sinogram_sample.txt) runs the demo offline."
echo "For real (non-clinical) PET sinograms, the cleanest sources are:"
echo
echo "  * PETRIC challenge data (Interfile/STIR sinograms):"
echo "      https://github.com/SyneRBI/PETRIC"
echo "  * SIRF-Exercises (openly usable phantom PET/MR data + notebooks):"
echo "      https://github.com/SyneRBI/SIRF-Exercises"
echo "  * Siemens mMR phantom datasets via STIR/SIRF:"
echo "      https://github.com/SyneRBI/SIRF"
echo "  * TCIA PET collections (mostly reconstructed volumes, license varies):"
echo "      https://www.cancerimagingarchive.net/"
echo "  * OpenNEURO PET datasets:"
echo "      https://openneuro.org/"
echo
echo "Notes:"
echo "  - Respect each collection's license and de-identification terms."
echo "  - Reconstruction needs the RAW sinogram or list-mode data, not just the"
echo "    reconstructed image; PETRIC/SIRF are the most direct for that."
echo "  - This project's loader expects the simple text format in data/README.md."
echo "    A converter from Interfile is left as an exercise (see README)."
echo
echo "For a larger SYNTHETIC problem right now, run:"
echo "    python scripts/make_synthetic.py --N 64 --K 60 --D 91 --iters 40"
