#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.7 : Medical Image Segmentation (Deep Learning)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs + license
# notes, and NEVER bypasses credentials/registration. The real medical
# segmentation datasets below are large and mostly require registration and a
# data-use agreement, so this script only PRINTS instructions and links and
# defers to scripts/make_synthetic.py for an offline stand-in. The committed
# synthetic sample in data/sample/ is sufficient to build and run the demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.7 -- Medical Image Segmentation (Deep Learning)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project ships a SYNTHETIC sample (data/sample/volume_sample.txt)"
echo "and needs no download to run. To experiment with real labelled volumes, fetch"
echo "one of the public segmentation datasets below (respect each license / DUA):"
echo
echo "  * Medical Segmentation Decathlon  -- http://medicaldecathlon.com/"
echo "      10 tasks (brain, heart, liver, ...), ~2,500 volumes, NIfTI (.nii.gz)."
echo "      Open registration; CC-BY-SA. Pick a task (e.g. Task03_Liver)."
echo "  * TotalSegmentator training set   -- https://zenodo.org/record/6802614"
echo "      ~1,200 CT with 117 structure labels. CC-BY; large (~30+ GB)."
echo "  * KiTS23 kidney tumor challenge   -- https://kits-challenge.org/kits23/"
echo "      Kidney/tumor CT; requires challenge registration."
echo "  * BraTS brain tumor dataset       -- https://www.synapse.org/#!Synapse:syn27046444"
echo "      Multi-modal MRI; requires Synapse account + DUA. DO NOT bypass."
echo
echo "Real volumes are NIfTI/DICOM; loading them needs a NIfTI/ITK reader (this"
echo "teaching build uses a plain text volume so the loader stays readable -- see"
echo "data/README.md). For a larger SYNTHETIC volume instead, run:"
echo "    python scripts/make_synthetic.py --D 32 --H 48 --W 48 --radius 6.0"
echo
echo "Idempotent-fetch pattern to follow when wiring a real set:"
echo "    1) skip the download if the file already exists with the right SHA256"
echo "    2) print source URL + expected size + checksum before downloading"
echo "    3) for credentialed sets, print registration instructions ONLY"
