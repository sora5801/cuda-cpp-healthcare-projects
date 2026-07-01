#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 4.33 : Real-Time MRI Reconstruction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. Every real dynamic/cardiac MRI raw-
# k-space dataset for this project sits behind a challenge registration or data-use
# agreement, so this script only PRINTS the instructions and links, and points at
# make_synthetic.py for an offline stand-in. The committed data/sample/ already lets
# the demo run offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data"

echo "[download_data] Project 4.33 -- Real-Time MRI Reconstruction"
echo "[download_data] Target data dir: $DATA_DIR"
echo ""
echo "All real dynamic/cardiac MRI raw-k-space datasets require a challenge"
echo "registration or a data-use agreement and CANNOT be auto-downloaded. Register"
echo "with the provider, then export a radial (or re-gridded) k-space trajectory into"
echo "the text layout documented in data/README.md."
echo ""
echo "  CMRxRecon 2023 -- cardiac MRI reconstruction challenge (multi-coil k-space):"
echo "     https://cmrxrecon.github.io/"
echo ""
echo "  ACDC -- Automated Cardiac Diagnosis Challenge (cine cardiac MRI):"
echo "     https://www.creatis.insa-lyon.fr/Challenge/acdc/"
echo ""
echo "  OCMR -- open cardiovascular MRI raw data (incl. real-time free-breathing):"
echo "     https://ocmr.info/"
echo ""
echo "The committed tiny SYNTHETIC sample in data/sample/ is enough to run the demo."
echo "For a larger synthetic problem (more spokes / frames / a bigger grid), run:"
echo "    python scripts/make_synthetic.py --n 64 --spokes 128 --win 34 --frames 8"
echo ""
echo "When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip the download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY (never bypass)"
