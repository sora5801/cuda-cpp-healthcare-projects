#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.32 : Protein Folding Pathway Extraction (Transition Path Sampling)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. A research-grade TPS run needs all-atom MD
# trajectories (Anton/Shaw, GPCRmd) and protein structures (PDB); this project
# ships a SYNTHETIC 1-D teaching model instead, so there is no bulk download --
# this script only prints where the real inputs live and defers to
# scripts/make_synthetic.py for the offline parameter file.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.32 -- Protein Folding Pathway Extraction (TPS)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project is a SYNTHETIC 1-D teaching model: its only input is the"
echo "parameter file data/sample/tps_params.txt, already committed. There is no"
echo "bulk dataset to download. The demo runs fully offline."
echo
echo "To (re)generate or rescale the synthetic parameter file:"
echo "    python scripts/make_synthetic.py --shooters 16384 --barrier 6.0"
echo
echo "Real, research-grade TPS inputs (require accounts / external requests --"
echo "this script does NOT fetch or bypass them):"
echo "    * Anton / D. E. Shaw millisecond folding trajectories (request access)"
echo "    * GPCRmd MD trajectories & pathways:  https://gpcrmd.org"
echo "    * Protein structures (Trp-cage 1L2Y, chignolin 5AWL):  https://www.rcsb.org"
echo "    * SAMPL host-guest kinetics challenges (search 'SAMPL challenge')"
echo
echo "Production TPS engines to study (do not copy wholesale):"
echo "    OpenPathSampling, WESTPA, HTMD -- see README 'Prior art & further reading'."
