#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.18 -- Image-Based 3D Printing / Model Generation for Surgery
#
# CONTRACT (CLAUDE.md sec.8): idempotent, documented, prints source URLs + access
# notes, and NEVER bypasses credentials/registration. The real clinical CT
# collections require registration and/or forbid redistribution, so this script
# only prints pointers and defers to scripts/make_synthetic.py for the offline,
# analytically-verifiable stand-in that the demo actually uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.18 -- Image-Based 3D Printing / Model Generation for Surgery"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC sphere volume (data/sample/volume_sample.txt)"
echo "so the demo runs offline and its result is analytically verifiable. The real"
echo "clinical datasets below are optional and gated behind registration/license:"
echo
echo "  TCIA body CT collections      https://www.cancerimagingarchive.net/   (per-collection license)"
echo "  OsteoArthritis Initiative     https://nda.nih.gov/oai/                 (registration required)"
echo "  VerSe vertebral CT            https://github.com/anjany/verse          (open)"
echo "  TotalSegmentator dataset      https://zenodo.org/record/6802614        (CC BY)"
echo
echo "We do NOT auto-download credentialed data. To use a real volume, download it"
echo "yourself, resample to a regular grid, and write it in the text format in"
echo "data/README.md (nx ny nz spacing origin iso, then the samples)."
echo
echo "For a larger SYNTHETIC volume that needs no download, run:"
echo "    python scripts/make_synthetic.py --n 65 --radius 24"
