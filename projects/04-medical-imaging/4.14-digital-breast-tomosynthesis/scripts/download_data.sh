#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch / point to the FULL dataset (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 4.14 -- Digital Breast Tomosynthesis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real DBT/mammography datasets are
# credentialed or non-redistributable, so this script prints how to obtain them
# and defers to scripts/make_synthetic.py for an offline synthetic stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.14 -- Digital Breast Tomosynthesis"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed tiny sample (data/sample/dbt_sample.txt) is SYNTHETIC and is"
echo "all the demo needs -- no download required. Real DBT/mammography data:"
echo
echo "  * CBIS-DDSM (curated mammograms via TCIA, open):"
echo "      https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM"
echo "  * BCS-DBT (Duke tomosynthesis challenge, true DBT projections):"
echo "      https://bcs-dbt.grand-challenge.org/"
echo "  * VinDr-Mammo (PhysioNet, CREDENTIALED -- requires a signed DUA):"
echo "      https://physionet.org/content/vindr-mammo/1.0.0/"
echo "  * OPTIMAM / OMI-DB (access via ICR UK, CREDENTIALED)."
echo
echo "This script does NOT bypass any registration/credential wall. For the"
echo "credentialed sets, register at the link, accept the licence, and place the"
echo "files under data/ yourself. For a larger SYNTHETIC problem instead, run:"
echo "    python scripts/make_synthetic.py --img 128 --angles 21 --det 160"
