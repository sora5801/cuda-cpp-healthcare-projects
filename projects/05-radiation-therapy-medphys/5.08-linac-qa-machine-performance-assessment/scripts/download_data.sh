#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to the FULL/real data (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 5.8 -- Linac QA & Machine Performance Assessment
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. There is no single downloadable file
# this demo consumes directly -- real linac-QA data is machine/vendor-specific
# and often site-restricted -- so this script prints authoritative pointers and
# defers to scripts/make_synthetic.py for the offline stand-in the demo uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.8 -- Linac QA & Machine Performance Assessment"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This demo runs on a SYNTHETIC sample (data/sample/qa_planes_sample.txt)."
echo "No real dataset is fetched. Authoritative reference material:"
echo
echo "  AAPM TG-119  IMRT QA test plans   : https://www.aapm.org/pubs/reports/RPT_82.pdf"
echo "  AAPM TG-218  tolerance limits     : https://doi.org/10.1002/mp.12810"
echo "  OpenMedPhys / awesome-medphys     : https://github.com/jrkerns/awesome-medphys"
echo "  Pylinac (example EPID/log data)   : https://github.com/jrkerns/pylinac"
echo
echo "Respect every dataset license; credentialed sets require registration"
echo "(this script does NOT bypass it). Regenerate the offline sample with:"
echo "    python scripts/make_synthetic.py            # 24x24 planes"
echo "    python scripts/make_synthetic.py --nx 128 --ny 128   # larger synthetic"
