#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 7.1 -- Diagnostic Imaging Classifier   (template skeleton)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. Defers to
# scripts/make_synthetic.py for an offline stand-in when needed.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 7.1 -- Diagnostic Imaging Classifier"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# TODO(impl): fill in the real dataset fetch. Template only prints guidance.
echo "TODO(impl): no full dataset wired up yet for this template skeleton."
echo "  Catalog dataset notes:"
echo "    MIMIC-CXR — 227,827 labelled chest X-ray studies with radiology reports from Beth Israel Deaconess (https://physionet.org/content/mimic-cxr/) CheXpert — 224,316 chest X-rays from Stanford, 14 pathology labels (https://stanfordmlgroup.github.io/competitions/chexpert/) LIDC-IDRI — 1,018 CT lung nodule cases with radiologist consensus annotations (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) The Cancer Imaging Archive (TCIA) — multi-modal oncology imaging across dozens of curated collections (https://www.cancerimagingarchive.net/)"
echo
echo "  The committed tiny sample in data/sample/ is enough to run the demo."
echo "  For a larger SYNTHETIC problem, run:"
echo "    python scripts/make_synthetic.py --n 1048576"
echo
echo "  When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
