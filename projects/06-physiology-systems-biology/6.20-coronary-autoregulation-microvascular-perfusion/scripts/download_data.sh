#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.20 -- Coronary Autoregulation & Microvascular Perfusion
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs + guidance,
# and NEVER bypasses credentials/registration. Every real coronary dataset here
# is credentialed or must be converted from a geometry model, so this script
# prints instructions and links ONLY and defers to make_synthetic.py for an
# offline stand-in. The committed tiny synthetic sample already runs the demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.20 -- Coronary Autoregulation & Microvascular Perfusion"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a TINY SYNTHETIC sample (data/sample/coronary_network.txt)"
echo "that runs the demo offline. There is no automatic bulk download because every"
echo "real coronary dataset is credentialed or must be converted from a geometry model."
echo
echo "Real datasets (obtain manually, respecting each license):"
echo "  * UK Biobank coronary CTA (subset)  https://www.ukbiobank.ac.uk"
echo "      -> requires an APPROVED application; redistribution forbidden."
echo "  * PhysioNet coronary pressure/flow   https://physionet.org"
echo "      -> some sets need credentialing + a data use agreement."
echo "  * Vascular Model Repository          http://www.vascularmodel.com"
echo "      -> open cardiovascular geometries; extract centerlines + radii."
echo "  * MICCAI coronary artery tracking     https://grand-challenge.org"
echo "      -> challenge registration required."
echo
echo "To build a REAL network for this solver:"
echo "  1) take a centerline model (nodes = branch points, edges = segments),"
echo "  2) write it into data/sample/coronary_network.txt in the documented format"
echo "     (see data/README.md), pinning the inlet and venous outlets,"
echo "  3) run the demo / exe on that file path."
echo
echo "For a larger SYNTHETIC network right now, regenerate the sample:"
echo "    python scripts/make_synthetic.py"
echo
echo "[download_data] No credentialed data was fetched or bypassed (by design)."
