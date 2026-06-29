#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.27 : MM-GBSA / MM-PBSA Rescoring
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs + notes,
# and NEVER bypasses credentials/registration. The committed tiny sample already
# lets the demo run offline; this script points at the real, credentialed
# datasets and defers to make_synthetic.py for a larger offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.27 -- MM-GBSA / MM-PBSA Rescoring"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project's REAL inputs are MD trajectories of protein-ligand complexes"
echo "plus a force field (charges, LJ params, Born radii). Those sources require"
echo "registration and carry licenses, so this script does NOT auto-download them."
echo
echo "  * PDBbind (complexes + measured affinities) : http://www.pdbbind.org.cn"
echo "  * CASF-2016 (scoring benchmark / core set)   : http://www.pdbbind.org.cn/casf.php"
echo "  * ChEMBL (bioactivity data)                  : https://www.ebi.ac.uk/chembl/"
echo "  * AMBER MM-GBSA tutorials (ready trajectories): https://ambermd.org/tutorials/"
echo
echo "  Respect each dataset's license; for credentialed sets, register at the URL"
echo "  above -- this script will not bypass any login."
echo
echo "  The committed sample (data/sample/complex_sample.txt) runs the demo offline."
echo "  For a LARGER synthetic problem (e.g. 64 snapshots), run:"
echo "    python scripts/make_synthetic.py --snapshots 64"
