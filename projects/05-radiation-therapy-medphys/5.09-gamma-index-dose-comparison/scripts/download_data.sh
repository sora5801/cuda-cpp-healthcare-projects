#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 5.9 -- Gamma-Index Dose Comparison
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, and NEVER bypasses
# credentials/registration. Real gamma-index inputs are patient-derived
# plan+measurement dose pairs that are not redistributable, so this script only
# prints guidance and defers to scripts/make_synthetic.py for an offline
# stand-in. There is nothing to download automatically.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.9 -- Gamma-Index Dose Comparison"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "No auto-download: gamma-index inputs are patient-derived plan+measurement"
echo "dose pairs and are not redistributable. Obtain them from your own clinic:"
echo
echo "  * AAPM TG-218  -- methodology + example criteria for IMRT/VMAT QA"
echo "                    (Med Phys 45(4), 2018)."
echo "  * Plan+measurement DICOM-RTDOSE pairs -- from your TPS + QA system"
echo "                    (film / EPID / diode array). Local access + ethics only."
echo "  * IROC-Houston phantom datasets -- request through IROC."
echo "  * Linac EPID measurement datasets -- from your machine's QA archive."
echo
echo "Respect every source license and patient-privacy rule. Do NOT commit"
echo "patient-derived data to this public repo."
echo
echo "The committed SYNTHETIC sample in data/sample/dose_pair.txt is enough to"
echo "run the demo. For a larger synthetic problem, run:"
echo "    python scripts/make_synthetic.py --n 128"
