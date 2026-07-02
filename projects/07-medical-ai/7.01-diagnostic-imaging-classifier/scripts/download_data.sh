#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 7.1 : Diagnostic Imaging Classifier   (reduced-scope teaching CNN)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + access
# notes, and NEVER bypasses credentials/registration. The real datasets require
# registration and forbid casual redistribution, so this script prints
# instructions + links and defers to make_synthetic.py for the offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 7.1 -- Diagnostic Imaging Classifier"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project ships a SYNTHETIC sample (data/sample/imaging_sample.txt)"
echo "and needs no download to run the demo. The real datasets below are"
echo "CREDENTIALED / license-restricted -- fetch them yourself after agreeing to"
echo "their terms; this script will not bypass registration."
echo
echo "  MIMIC-CXR   (credentialed, PhysioNet DUA):"
echo "    https://physionet.org/content/mimic-cxr/"
echo "  CheXpert    (registration, research-use license):"
echo "    https://stanfordmlgroup.github.io/competitions/chexpert/"
echo "  LIDC-IDRI   (TCIA, confirm per-collection terms):"
echo "    https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI"
echo "  TCIA        (per-collection licenses):"
echo "    https://www.cancerimagingarchive.net/"
echo
echo "  To (re)generate the offline synthetic sample the demo uses:"
echo "    python scripts/make_synthetic.py"
echo
echo "  When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
