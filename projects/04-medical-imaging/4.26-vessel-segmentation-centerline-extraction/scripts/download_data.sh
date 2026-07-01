#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real vessel-imaging dataset pointers (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 4.26 : Vessel Segmentation & Centerline Extraction
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints the source URLs,
# and NEVER bypasses credentials/registration. The real vessel datasets need an
# account or challenge sign-up, so this script only prints instructions + links;
# scripts/make_synthetic.py provides the offline stand-in the demo uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.26 -- Vessel Segmentation & Centerline Extraction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "There is nothing to auto-download: the committed synthetic volume in"
echo "data/sample/vessel_volume.txt is enough to run the demo, and the real"
echo "datasets below need registration (do NOT try to bypass it)."
echo
echo "REAL 3-D vessel datasets (register on the site, then export to NIfTI):"
echo "  ASOCA (coronary CTA challenge) : https://asoca.grand-challenge.org/"
echo "  ImageCAS (1000 coronary CTAs)  : https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT"
echo "  3D-IRCADb-01 (abdominal/liver) : https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/"
echo
echo "To run this teaching filter on real data you must first convert a NIfTI/"
echo "DICOM volume into this project's plain-text format (see data/README.md)."
echo "A tiny converter is left as an exercise (README 'Exercises')."
echo
echo "Bigger SYNTHETIC volume (no download, fully offline):"
echo "  python scripts/make_synthetic.py --nx 128 --ny 96 --nz 96 --radius 4"
