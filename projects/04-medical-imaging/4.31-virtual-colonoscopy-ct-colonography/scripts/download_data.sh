#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Guidance for the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.31 : Virtual Colonoscopy & CT Colonography
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# guidance, and NEVER bypasses credentials/registration. Real CT colonography
# volumes require accepting data-use terms, so this script only PRINTS where to
# get them and defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.31 -- Virtual Colonoscopy & CT Colonography"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/colon_volume_sample.txt) is SYNTHETIC"
echo "and is all the demo needs -- it runs fully offline."
echo
echo "To work with REAL CT colonography volumes (not auto-downloaded here):"
echo
echo "  TCIA 'CT Colonography' collection (supine/prone DICOM):"
echo "    https://wiki.cancerimagingarchive.net/display/Public/CT+Colonography"
echo "    - Accept the TCIA Data Usage Policy / per-collection terms."
echo "    - Download with the NBIA Data Retriever (a manifest-based tool)."
echo "    - Then segment the air-filled lumen and resample to a dense grid"
echo "      in the loader's text format (see data/README.md)."
echo
echo "  Other sources (may require registration): MICCAI 2018 colon challenge,"
echo "  ACR Lung-RADS CT, NLST CT colonography subsets."
echo
echo "For a larger SYNTHETIC volume instead, run:"
echo "    python scripts/make_synthetic.py --nx 96 --ny 96 --nz 128 --width 256 --height 256"
echo
echo "This script intentionally downloads nothing: the public CTC sets are"
echo "credentialed/terms-gated and must be fetched by you, per their license."
