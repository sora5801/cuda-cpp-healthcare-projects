#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real multi-site imaging pointers (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 4.25 : Image Harmonization Across Scanners/Sites.
#
# There is NOTHING to auto-download: every real multi-site imaging dataset below
# requires registration / a data-use agreement, and most FORBID redistribution.
# This script NEVER attempts to bypass credentials (CLAUDE.md §8). It prints the
# official links and how to turn extracted features into our loader format; the
# committed synthetic sample lets the demo run offline in the meantime.
# ===========================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 4.25 -- Image Harmonization Across Scanners/Sites"
echo
echo "ComBat operates on EXTRACTED FEATURES (e.g. FreeSurfer regional volumes /"
echo "cortical thickness, or radiomic features), not raw voxels. Export a feature"
echo "table into the format in data/README.md:"
echo "    line 1 : N P B C"
echo "    line 2 : batch (scanner/site) label per sample"
echo "    N lines: C covariate values per sample (age, sex, ...)"
echo "    P lines: N feature values per line"
echo
echo "Public multi-site imaging datasets (registration / DUA required):"
echo "  ABIDE (autism, multi-site)  : http://fcon_1000.projects.nitrc.org/indi/abide/"
echo "  ADNI  (Alzheimer's)         : https://adni.loni.usc.edu/"
echo "  IXI   (multi-site brain MRI): https://brain-development.org/ixi-dataset/"
echo "  UK Biobank imaging          : https://www.ukbiobank.ac.uk/"
echo
echo "Reference implementation to compare against:"
echo "  NeuroComBat : https://github.com/Jfortin1/ComBatHarmonization"
echo
echo "No download needed -- generate a bigger SYNTHETIC set instead:"
echo "  python scripts/make_synthetic.py --p 200 --b 4 --n 120"
echo
echo "Target data dir: $ROOT/data"
