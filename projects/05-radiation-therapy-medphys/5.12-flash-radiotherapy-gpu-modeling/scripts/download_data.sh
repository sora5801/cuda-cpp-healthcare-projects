#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 5.12 : FLASH Radiotherapy GPU Modeling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. This project's "input" is a tiny
# ensemble-configuration file (a parameter sweep); the physics lives in the
# code, so there is NO large binary dataset to download. Real FLASH-RT
# validation data (dosimetry, tumour oxygenation, radiolysis yields) is
# credentialed or not redistributable -- we point to it and generate a
# synthetic stand-in instead.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 5.12 -- FLASH Radiotherapy GPU Modeling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC ensemble-config sample; no bulk download is"
echo "needed to run the demo. Real-world reference data (for those extending the"
echo "model) is credentialed / not redistributable:"
echo
echo "  * FLASH-RT experimental dosimetry -- CERN/CLEAR, UCLouvain, Stanford FLASH"
echo "    programs (institutional access; verify each program's data-sharing policy)."
echo "  * AAPM FLASH-RT working-group benchmark datasets (verify current URL)."
echo "  * Published tumour oxygen-tension (pO2) measurements -- see the literature"
echo "    (e.g. Eppendorf-electrode and EPR-oximetry studies)."
echo "  * Geant4-DNA radiolysis validation datasets -- https://geant4-dna.org"
echo
echo "To (re)generate the committed synthetic sample offline, run:"
echo "    python scripts/make_synthetic.py"
echo
echo "For a finer oxygen sweep (more ensemble members), run e.g.:"
echo "    python scripts/make_synthetic.py --n-po2 32"
echo
echo "[download_data] Nothing to download; done."
