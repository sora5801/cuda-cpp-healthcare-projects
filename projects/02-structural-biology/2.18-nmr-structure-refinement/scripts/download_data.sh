#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.18 : NMR Structure Refinement
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. NMR restraint data is assembled
# per-protein-entry (no single archive), so this script prints the canonical
# sources and defers to scripts/make_synthetic.py for the offline demo input.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.18 -- NMR Structure Refinement"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a SYNTHETIC committed sample (data/sample/restraints.txt),"
echo "which is all the demo needs. There is no single 'full dataset' to download:"
echo "real NMR restraints are assembled per protein entry from these sources:"
echo
echo "  * BMRB  (https://bmrb.io)        - assigned shifts + restraint lists"
echo "  * PDB   (https://www.rcsb.org)   - deposited NMR model ensembles + restraints"
echo "  * RECOORD                        - uniformly recalculated NMR structures"
echo "  * CASD-NMR                       - blind structure-determination benchmarks"
echo
echo "  None of the above is fetched automatically; visit the site for the entry"
echo "  you want and respect its terms of use. To experiment at larger scale with"
echo "  no download, regenerate a bigger SYNTHETIC problem:"
echo "    python scripts/make_synthetic.py --n-beads 24 --replicas 4096"
echo
echo "  If you later wire a real fetch, follow the idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
