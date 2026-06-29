#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch / point at the FULL dataset (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 1.1 : Molecular Dynamics Engine  (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This teaching engine runs on a SYNTHETIC
# Lennard-Jones fluid (data/sample/lj_sample.txt) and needs no external download,
# so this script just ensures the synthetic sample exists and prints pointers to
# the real force fields a production engine would consume.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
SAMPLE="$DATA_DIR/sample/lj_sample.txt"

echo "[download_data] Project 1.1 -- Molecular Dynamics Engine"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# (a) Idempotent: regenerate the tiny synthetic sample only if it is missing.
if [[ -f "$SAMPLE" ]]; then
  echo "[download_data] Synthetic sample already present: $SAMPLE"
else
  echo "[download_data] Generating synthetic sample ..."
  python "$(dirname "${BASH_SOURCE[0]}")/make_synthetic.py"
fi

# (b) Pointers to the real force fields / trajectory libraries (study material).
echo
echo "This engine is a teaching model of the Lennard-Jones force field, so it"
echo "runs entirely on the committed SYNTHETIC sample -- no download required."
echo
echo "Production biomolecular MD instead reads these (do NOT commit them here):"
echo "  CHARMM36m force field  : https://mackerell.umaryland.edu/charmm_ff.shtml"
echo "  AMBER ff19SB           : https://ambermd.org"
echo "  GPCRmd trajectories    : https://gpcrmd.org"
echo "  MoDEL protein library  : https://mmb.irbbarcelona.org/MoDEL/"
echo
echo "For a larger SYNTHETIC system (e.g. 512 atoms), run:"
echo "    python scripts/make_synthetic.py --side 8"
