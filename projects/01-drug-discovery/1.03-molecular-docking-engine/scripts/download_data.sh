#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.3 -- Molecular Docking Engine   (template skeleton)
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

echo "[download_data] Project 1.3 -- Molecular Docking Engine"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# This teaching project runs on SYNTHETIC input (scripts/make_synthetic.py). Real
# docking data is not auto-fetched: it requires receptor/ligand preparation and an
# AutoGrid map computation that are outside this didactic scope. We print pointers.
echo "  This project's committed sample is SYNTHETIC (data/sample/) and runs the demo offline."
echo "  No real dataset is auto-downloaded -- real docking needs receptor + ligand prep."
echo
echo "  Real datasets to study (respect each license):"
echo "    DUD-E    102 targets, actives + decoys      https://dude.docking.org"
echo "    ChEMBL   >2M bioactive compounds            https://www.ebi.ac.uk/chembl/"
echo "    PDBbind  protein-ligand complexes + Kd/Ki   http://www.pdbbind.org.cn"
echo "    CASF     scoring-function benchmark         http://www.pdbbind.org.cn/casf.php"
echo
echo "  To dock a real complex you would: prepare receptor+ligand to PDBQT (AutoDockTools/Meeko),"
echo "  precompute energy maps with AutoGrid, then run AutoDock-GPU or Vina (see THEORY.md)."
echo
echo "  For a larger SYNTHETIC problem instead, run e.g.:"
echo "    python scripts/make_synthetic.py --n-trans 15 --n-rot 6 --n-grid 32"
