#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.21 : Microcirculation & Oxygen Transport
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. Defers to
# scripts/make_synthetic.py for an offline stand-in when needed.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.21 -- Microcirculation & Oxygen Transport"
echo "[download_data] Target data dir: $DATA_DIR"
echo

echo "This project ships a SYNTHETIC sample (data/sample/microvessel_network.txt)"
echo "and does not require any download to run the demo. Real microvascular data:"
echo "  - Vascular Model Repository            : http://www.vascularmodel.com"
echo "  - Allen Institute two-photon microscopy: https://portal.brain-map.org"
echo "  - PhysioNet O2 saturation waveforms    : https://physionet.org (credentialed)"
echo "  - Secomb-group microvascular networks  : https://secomb.org (verify terms)"
echo "Respect each dataset's license/registration; this script never bypasses it."
echo
echo "  For a larger SYNTHETIC problem, run:"
echo "    python scripts/make_synthetic.py --nx 24 --ny 24 --nz 16"
echo
echo "  When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
