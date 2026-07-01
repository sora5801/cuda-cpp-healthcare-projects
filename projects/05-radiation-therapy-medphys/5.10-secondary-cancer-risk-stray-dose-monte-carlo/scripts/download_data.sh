#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 5.10 : Secondary Cancer Risk & Stray-Dose Monte Carlo
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. The real inputs
# (ICRP-110 voxel phantoms, NIST XCOM cross-sections, TCIA CTs) are large and/or
# registration-gated, so this script prints instructions + links and defers to
# scripts/make_synthetic.py for the offline stand-in the demo uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.10 -- Secondary Cancer Risk & Stray-Dose Monte Carlo"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC committed sample (data/sample/phantom.txt),"
echo "so no download is required to run the demo."
echo
echo "Real datasets (registration/attribution required -- fetch by hand):"
echo "  * ICRP 110 voxel phantoms (adult male/female):"
echo "      https://www.icrp.org/publication.asp?id=ICRP%20Publication%20110"
echo "  * NIST XCOM photon cross-sections:"
echo "      https://www.nist.gov/pml/xcom-photon-cross-sections"
echo "  * TCIA proton-therapy planning CTs (account + attribution):"
echo "      https://www.cancerimagingarchive.net/"
echo
echo "For a larger SYNTHETIC problem instead, run:"
echo "    python scripts/make_synthetic.py --histories 2000000 --seed 7"
