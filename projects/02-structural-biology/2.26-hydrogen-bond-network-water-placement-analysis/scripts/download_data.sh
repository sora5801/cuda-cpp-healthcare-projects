#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.26 -- Hydrogen Bond Network & Water Placement Analysis
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. GIST on real
# structures needs a full MD trajectory + a real GIST tool, outside this teaching
# project's scope -- so this script prints the authoritative pointers and defers
# to scripts/make_synthetic.py for the offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
SAMPLE="$DATA_DIR/sample/water_sample.txt"

echo "[download_data] Project 2.26 -- Hydrogen Bond Network & Water Placement Analysis"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# Idempotent: the committed synthetic sample is all the demo needs. If missing,
# regenerate it deterministically rather than downloading anything.
if [[ -f "$SAMPLE" ]]; then
  echo "[download_data] Synthetic sample already present: $SAMPLE"
else
  echo "[download_data] Synthetic sample missing; regenerating it ..."
  python "$(dirname "${BASH_SOURCE[0]}")/make_synthetic.py"
fi

echo
echo "[download_data] This project ships a SYNTHETIC sample (see data/README.md)."
echo "  No real dataset is required to build, run, or study the demo."
echo
echo "  To study GIST on REAL structures, use these public sources (respect each license;"
echo "  do NOT commit redistributed data; nothing here is for clinical use):"
echo "    * SAMPL water-placement challenges : https://github.com/samplchallenges/SAMPL"
echo "    * Explicit-solvent PDB structures  : https://www.rcsb.org"
echo "    * GIST reference systems           : T4 lysozyme L99A, FKBP12 (GIST literature)"
echo "    * WaterMap validation sets         : Schrodinger (commercial; verify URL)"
echo
echo "  Producing a real GIST input requires an MD trajectory (AMBER/GROMACS/OpenMM) and"
echo "  a GIST tool (cpptraj 'gist' or GISTPP). When wiring such a fetch, follow the"
echo "  idempotent pattern: (1) skip if the file exists with the right SHA256,"
echo "  (2) print source URL + expected size + checksum, (3) for credentialed sets print"
echo "  registration instructions ONLY -- never bypass them."
echo
echo "  For a LARGER synthetic problem instead:"
echo "    python scripts/make_synthetic.py --frames 5000"
