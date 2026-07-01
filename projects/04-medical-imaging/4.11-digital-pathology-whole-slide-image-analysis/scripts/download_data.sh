#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real WSI-data pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.11 : Digital Pathology / Whole-Slide Image Analysis
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. Whole-slide images are multi-gigabyte
# and their public repositories require a (free) account and a data-use
# agreement, so this script does NOT auto-download; it prints where to get the
# data and how to turn it into this project's tile-feature-bag format. The
# committed synthetic sample already lets the demo run offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.11 -- Digital Pathology / Whole-Slide Image Analysis"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project consumes a BAG of tile FEATURE vectors per slide, not raw"
echo "pixels. The real pipeline is: download WSIs -> tile + tissue-detect ->"
echo "run a frozen CNN/ViT encoder per tile -> save the N x D features."
echo
echo "Public WSI datasets (free account + data-use agreement required):"
echo "  TCGA slides (GDC) : https://portal.gdc.cancer.gov/"
echo "  CAMELYON16/17     : https://camelyon17.grand-challenge.org/"
echo "  TUPAC16           : http://tupac.tue-image.nl/"
echo
echo "Tools to read WSIs and extract features:"
echo "  OpenSlide         : https://openslide.org/            (read .svs/.tif pyramids)"
echo "  CLAM              : https://github.com/mahmoodlab/CLAM (tiling + feature bags + MIL)"
echo "  UNI encoder       : https://github.com/mahmoodlab/UNI  (pretrained ViT features)"
echo
echo "Export each slide as 'N D label' then N rows of D features (D must equal"
echo "FEAT_DIM in src/wsi.h). See data/README.md for the exact format."
echo
echo "No download needed to run the demo. Bigger SYNTHETIC bag:"
echo "  python scripts/make_synthetic.py --n 20000"
