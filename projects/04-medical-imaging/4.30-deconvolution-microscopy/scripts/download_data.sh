#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.30 : Deconvolution Microscopy
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Public microscopy benchmark sets are
# large TIFF stacks under their own licenses; we do not redistribute them. The
# committed tiny SYNTHETIC sample (data/sample/) makes the demo run offline; for
# a bigger synthetic image use scripts/make_synthetic.py.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.30 -- Deconvolution Microscopy"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a TINY SYNTHETIC blurred image in data/sample/ so the"
echo "demo runs fully offline. No download is required to build, run, or learn."
echo
echo "To study REAL fluorescence-microscopy deconvolution benchmarks, visit:"
echo "  * EPFL Biomedical Imaging Group deconvolution benchmark + measured PSFs:"
echo "      https://bigwww.epfl.ch/deconvolution/"
echo "  * BioImage Archive fluorescence datasets (raw + restored stacks):"
echo "      https://www.ebi.ac.uk/biostudies/bioimages"
echo "  * ImageJ/Fiji sample images (e.g. the classic confocal stacks):"
echo "      https://imagej.net/"
echo
echo "Each is governed by its own license -- respect it; we do not redistribute."
echo "Convert a downloaded 2-D slice to this project's text format:"
echo "  header line '<w> <h>' then h rows of w space-separated intensities,"
echo "  matching load_image() in src/reference_cpu.cpp (see data/README.md)."
echo
echo "For a larger SYNTHETIC image (no download), run:"
echo "  python scripts/make_synthetic.py --w 128 --h 128"
