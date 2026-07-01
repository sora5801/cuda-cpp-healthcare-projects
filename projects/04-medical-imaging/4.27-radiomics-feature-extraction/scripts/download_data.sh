#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.27 : Radiomics Feature Extraction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL, and
# NEVER bypasses credentials/registration. Real radiomics data is large DICOM
# imaging with segmentation masks (TCIA, GDC); those must not be redistributed
# here, so this script only prints instructions + links and defers to
# scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.27 -- Radiomics Feature Extraction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed tiny SYNTHETIC sample (data/sample/radiomics_sample.txt) is"
echo "enough to build and run the demo offline. No download is required."
echo
echo "Real radiomics cohorts (imaging + segmentations) live at:"
echo "  * TCIA NSCLC-Radiomics (422 lung CTs + survival):"
echo "      https://www.cancerimagingarchive.net/collection/nsclc-radiomics/"
echo "  * QIN-HEADNECK (head & neck RT), RIDER Breast MRI -- via TCIA."
echo "  * TCGA imaging + clinical: https://portal.gdc.cancer.gov/"
echo
echo "How to use real data with this project:"
echo "  1) Download a collection from TCIA (respect each collection's LICENSE;"
echo "     some require a Data Use Agreement -- do NOT bypass it)."
echo "  2) Read the CT/PET/MRI volume and its ROI segmentation with pydicom /"
echo "     SimpleITK, crop to the ROI bounding box, quantize intensities to Ng"
echo "     levels, and write the 'nx ny nz Ng' + intensities + mask text format"
echo "     that data/README.md documents (this conversion is an exercise)."
echo
echo "For a larger SYNTHETIC problem instead, run:"
echo "    python scripts/make_synthetic.py --nx 64 --ny 64 --nz 48 --ng 16"
