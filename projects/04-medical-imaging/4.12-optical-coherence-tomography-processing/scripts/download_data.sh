#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real OCT dataset pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.12 : Optical Coherence Tomography Processing (SD-OCT reconstruction)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The public OCT datasets below ship PROCESSED
# B-scan images (already reconstructed), not the vendor RAW spectra this project's
# reconstruction consumes -- so nothing is downloaded here; we point at the
# datasets for downstream tasks and defer to scripts/make_synthetic.py for a
# runnable RAW-spectrum stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.12 -- Optical Coherence Tomography Processing"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Public OCT datasets (PROCESSED B-scans / volumes, for segmentation/classification):"
echo "  OCTDL     : https://www.nature.com/articles/s41597-024-03182-7  (2,064 labeled B-scans)"
echo "  Duke DME  : https://people.duke.edu/~sf59/Chiu_BOE_2012_dataset.htm  (110 annotated volumes)"
echo "  OCTA-500  : https://arxiv.org/abs/2012.07261  (OCT angiography volumes with labels)"
echo
echo "NOTE: those provide reconstructed images, not vendor RAW spectra. This project"
echo "reconstructs FROM raw spectra, so the committed sample is synthetic raw"
echo "interferograms (scripts/make_synthetic.py). Raw-spectrum access requires the"
echo "OCT device SDK (Thorlabs/Bioptigen/Heidelberg) -- follow the vendor's terms;"
echo "this script will not bypass any registration."
echo
echo "Bigger SYNTHETIC B-scan (no download):"
echo "  python scripts/make_synthetic.py --n-ascan 128 --n-spec 1024"
