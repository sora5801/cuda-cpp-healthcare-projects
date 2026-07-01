#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 5.4 : Collapsed-Cone / Superposition-Convolution Dose
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The real dose-engine benchmark sets
# below all require registration or forbid redistribution, so this script only
# prints instructions + links and defers to scripts/make_synthetic.py for the
# offline stand-in that the demo actually uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.4 -- Collapsed-Cone / Superposition-Convolution Dose"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a TINY SYNTHETIC phantom (data/sample/phantom.txt) that is"
echo "enough to run the demo offline. The reference benchmark datasets below are for"
echo "going further; each requires registration or has redistribution limits, so we"
echo "do NOT download them automatically -- follow the links and accept each license."
echo
echo "  * AAPM TG-105 report + heterogeneous-media dose test cases:"
echo "      https://www.aapm.org/pubs/reports/  (search 'TG-105')"
echo "  * IROC Houston phantom credentialing (lung phantom CT + dosimetry):"
echo "      https://www.mdanderson.org/  (IROC Houston Quality Assurance Center)"
echo "  * TCIA clinical photon planning datasets (CT + RTDOSE/RTPLAN DICOM):"
echo "      https://www.cancerimagingarchive.net/  (register, then browse RT collections)"
echo "  * CIRS IMRT verification phantom data: https://www.cirsinc.com/"
echo
echo "To regenerate the committed SYNTHETIC phantom instead:"
echo "    python scripts/make_synthetic.py"
echo
echo "When wiring a real dataset later, keep this idempotent pattern:"
echo "    1) skip the download if the file already exists with the right SHA256"
echo "    2) print source URL + expected size + checksum before fetching"
echo "    3) for credentialed sets, print registration instructions ONLY"
