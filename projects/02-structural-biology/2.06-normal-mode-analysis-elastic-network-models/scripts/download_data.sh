#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real protein-structure pointers (Linux/macOS)
# Project 2.06 : Normal Mode Analysis / Elastic Network Models. Nothing to fetch.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 2.06 -- Normal Mode Analysis / Elastic Network Models"
echo
echo "Use a real structure: download a PDB/CIF, extract the CA atoms' x/y/z,"
echo "and prepend 'N cutoff' to make data/sample/protein_ca.txt."
echo
echo "  RCSB PDB     : https://www.rcsb.org           (experimental structures)"
echo "  AlphaFold DB : https://alphafold.ebi.ac.uk    (predicted structures)"
echo "  ProDy        : https://github.com/prody/ProDy  (ANM/GNM; parses PDB)"
echo
echo "Tip (with ProDy): prody.parsePDB('1abc').select('name CA').getCoords()"
echo
echo "Bigger synthetic structure (no download):"
echo "  python scripts/make_synthetic.py --N 120"
echo
echo "Target data dir: $PROJECT_ROOT/data"
