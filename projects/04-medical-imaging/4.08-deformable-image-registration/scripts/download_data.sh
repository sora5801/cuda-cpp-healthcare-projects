#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real DIR dataset pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.8 : Deformable Image Registration (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, and it NEVER bypasses
# credentials or registration. This project ships its own SYNTHETIC image pair,
# so there is nothing to auto-download; the real registration datasets below all
# require agreeing to a data-use / challenge license, which you must do yourself.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.8 -- Deformable Image Registration"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "There is NOTHING to download for the demo: this project generates its"
echo "own synthetic fixed/moving image pair (data/sample/dir_pair.txt)."
echo
echo "REAL registration benchmarks (each needs you to register/accept a license):"
echo "  Learn2Reg challenge : https://learn2reg.grand-challenge.org/"
echo "      lung / brain / abdominal CT + MR pairs with evaluation."
echo "  OASIS brain MRI     : https://www.oasis-brains.org/"
echo "      the brain set used by the Learn2Reg inter-subject task."
echo "  DIR-Lab lung CT     : https://dir-lab.com/"
echo "      4D-CT respiratory pairs with expert landmarks (gold-standard TRE)."
echo
echo "Do NOT commit any of the above into this repo (license + patient data)."
echo "Convert a real image slice into this project's text format yourself, or"
echo "make a bigger SYNTHETIC pair (no download, fully offline):"
echo "  python scripts/make_synthetic.py --nx 128 --ny 128 --shift 8.0"
