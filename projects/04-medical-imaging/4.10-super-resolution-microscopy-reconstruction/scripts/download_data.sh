#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real SMLM-data pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.10 : Super-Resolution Microscopy Reconstruction  (SMLM). Nothing to
# fetch automatically -- real STORM/PALM movies are large multi-GB TIFF stacks
# and several need registration, so per CLAUDE.md §8 this script only prints the
# sources and defers to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.10 -- Super-Resolution Microscopy Reconstruction"
echo
echo "Real SMLM movies are multi-frame TIFF/OME-TIFF stacks. To use one, export"
echo "each frame's pixels into the text format in data/README.md:"
echo "  header:  'F H W background threshold'   then F*H*W floats, row-major."
echo
echo "  EPFL SMLM Challenge : https://srm.epfl.ch/srm/dataset/challenge-2016/"
echo "                        (synthetic + real STORM/PALM frames, with ground truth)"
echo "  BioImage Archive    : https://www.ebi.ac.uk/biostudies/bioimages"
echo "                        (public SMLM collections)"
echo "  OME-TIFF standard   : https://www.openmicroscopy.org/ome-files/"
echo
echo "Tools to read/convert TIFF stacks: tifffile (Python), Fiji/ImageJ, ThunderSTORM."
echo
echo "No download needed for the demo -- the committed data/sample/smlm_stack.txt"
echo "runs offline. For a bigger SYNTHETIC movie:"
echo "    python scripts/make_synthetic.py --frames 200 --width 64 --height 64"
echo
echo "Target data dir: $DATA_DIR"
