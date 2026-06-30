#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.4 : Cryo-ET Subtomogram Averaging
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs and NEVER
# bypasses credentials/registration. Real cryo-ET subtomograms are large and
# have their own usage/citation policies, so this script prints links and
# instructions ONLY and defers to scripts/make_synthetic.py for the offline
# stand-in that the demo actually uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.4 -- Cryo-ET Subtomogram Averaging"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC committed sample (data/sample/), so the demo"
echo "runs offline with no download. Real cryo-ET data lives here:"
echo
echo "  EMDB STA maps       : https://www.ebi.ac.uk/emdb/"
echo "  EMPIAR raw data     : https://www.ebi.ac.uk/empiar/  (e.g. EMPIAR-10064)"
echo "  SHREC cryo-ET bench : search 'SHREC subtomogram challenge' (URL moves yearly)"
echo "  CryoDRGN-ET         : https://github.com/ml-struct-bio/cryodrgn"
echo
echo "These sets are large (GBs-TBs) and have their own citation/usage terms;"
echo "this repo does NOT redistribute them. Respect each source's license."
echo
echo "To (re)generate the synthetic sample the demo uses:"
echo "    python scripts/make_synthetic.py            # default: 6 cubes, 16^3, 12 angles"
echo "    python scripts/make_synthetic.py --d 32     # a bigger synthetic problem"
echo
echo "When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip the download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY (never bypass)"
echo "    4) extract d^3 cubes around picked particles into the loader's text layout"
echo "       (header 'n_sub d n_angles', then the reference cube, then candidate cubes)"
