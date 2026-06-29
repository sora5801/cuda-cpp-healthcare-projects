#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to the FULL datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.33 -- Interaction Fingerprinting & Binding-Mode Clustering
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real interaction fingerprints come from
# docking poses or MD frames of actual complexes; turning those into the bit-
# vectors this project clusters needs a chemistry toolkit (ProLIF/ODDT), which is
# out of scope. So this script PRINTS where to get the structures + how to derive
# IFPs, and defers to make_synthetic.py for an offline, self-contained stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.33 -- Interaction Fingerprinting & Binding-Mode Clustering"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project clusters INTERACTION FINGERPRINTS (bit-vectors). To build them"
echo "from real structures you need (a) protein-ligand poses and (b) an interaction"
echo "detector. Public sources for the structures:"
echo "  * PDBbind   - complexes + affinities : http://www.pdbbind.org.cn"
echo "  * KLIFS     - kinase IFP features    : https://klifs.net"
echo "  * ChEMBL    - bioactivity + structs  : https://www.ebi.ac.uk/chembl/"
echo "  * BindingDB - measured binding data  : https://www.bindingdb.org"
echo
echo "Derive IFPs with a toolkit, then emit rows matching data/README.md:"
echo "  * ProLIF : https://github.com/chemosim-lab/ProLIF  (IFPs from MD/poses)"
echo "  * ODDT   : https://github.com/oddt/oddt           (open drug-discovery toolkit)"
echo
echo "No credentials are bypassed here. The committed data/sample/ifp_sample.txt is"
echo "enough to run the demo offline. For a larger SYNTHETIC problem:"
echo "    python scripts/make_synthetic.py --per-mode 500"
