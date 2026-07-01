#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 5.3 : Proton & Heavy-Ion Therapy Dose
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL(s), and
# NEVER bypasses credentials/registration. The real proton-therapy datasets are
# either patient-derived (need an institutional/DUA agreement) or ship inside a
# Monte-Carlo toolkit (TOPAS/GATE benchmark beams). This project does NOT need
# any of them to run: the committed synthetic plan drives the demo. This script
# prints where to get real data and defers to make_synthetic.py.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.3 -- Proton & Heavy-Ion Therapy Dose"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project runs entirely on the committed SYNTHETIC plan in"
echo "data/sample/proton_plan_sample.txt -- no download is required."
echo
echo "Real / reference proton data you may study (respect every license):"
echo "  * TOPAS / GATE benchmark proton beams -- integral depth-dose and lateral"
echo "    profiles used to COMMISSION analytic engines. Obtain via the TOPAS"
echo "    (https://www.topasmc.org) or GATE (http://www.opengatecollaboration.org)"
echo "    distributions; these are Monte-Carlo outputs, not patient data."
echo "  * POPI-model 4D CT for treatment planning:"
echo "    https://www.creatis.insa-lyon.fr/rio/popi-model"
echo "  * TCIA proton treatment-response collections (registration/DUA required):"
echo "    https://www.cancerimagingarchive.net"
echo
echo "Patient-derived / credentialed sets: this script will NOT bypass any"
echo "registration or data-use agreement. Follow the provider's process."
echo
echo "To make a LARGER synthetic plan (e.g. a spread-out Bragg peak), run:"
echo "    python scripts/make_synthetic.py --ranges 8 9 10 11 12"
