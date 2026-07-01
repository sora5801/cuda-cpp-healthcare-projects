#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 5.13 : BNCT Dose Calculation & Optimization (reduced-scope teaching MC)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. This project's demo runs entirely on
# the committed SYNTHETIC sample (data/sample/bnct_params.txt); the real BNCT
# reference data below is optional and only relevant if you extend the model
# toward a validated code. We therefore print guidance rather than downloading
# license-restricted or registration-gated material.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.13 -- BNCT Dose Calculation & Optimization"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC parameter sample that fully drives the demo:"
echo "    $DATA_DIR/sample/bnct_params.txt"
echo "Regenerate or scale it with:"
echo "    python scripts/make_synthetic.py --histories 1000000 --seed 7"
echo
echo "OPTIONAL real BNCT references (for extending toward a validated code):"
echo "  * OpenMC (open-source, GPU-capable neutron Monte Carlo) + its validation"
echo "    tests:  https://github.com/openmc-dev/openmc/tree/develop/tests"
echo "  * GATE 10 neutron transport for BNCT: https://github.com/OpenGATE/opengate"
echo "  * ENDF/B-VIII.0 evaluated neutron cross sections (used by real codes):"
echo "    https://www.nndc.bnl.gov/endf/   (verify current URL; large, evaluated)"
echo "  * IAEA BNCT benchmark cases: search https://www.iaea.org (registration/"
echo "    request may be required -- this script will NOT bypass that)."
echo
echo "None of the above is required to run the demo. If you download ENDF/B or"
echo "IAEA data yourself, respect each source's license; do not redistribute"
echo "gated data through this repository (CLAUDE.md §8)."
echo
echo "Idempotent pattern to follow when wiring a real fetch:"
echo "    1) skip the download if the file already exists with the right SHA256"
echo "    2) print source URL + expected size + checksum before downloading"
echo "    3) for credentialed sets, print registration instructions ONLY"
