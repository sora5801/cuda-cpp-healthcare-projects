#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL fingerprints (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 1.12 : Molecular Fingerprint Similarity Search
#
# Prints the RDKit recipe; downloads nothing and needs no credentials. Use
# make_synthetic.py for an offline stand-in (CLAUDE.md section 8).
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 1.12 -- Molecular Fingerprint Similarity Search"
echo
echo "Real fingerprints are generated from a molecule library with RDKit:"
echo "  1) Get SMILES, e.g. ChEMBL (https://www.ebi.ac.uk/chembl/) or"
echo "     ZINC20 (https://zinc20.docking.org)."
echo "  2) pip install rdkit"
echo "  3) ECFP4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)"
echo "  4) Pack each 2048-bit vector into 32 little-endian uint64 words and write"
echo "     the hex format in data/README.md (1 query line + n library lines)."
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py --n 1000000"
echo
echo "Keep FP_WORDS (=32, i.e. 2048 bits) consistent with src/reference_cpu.h."
echo "Target data dir: $PROJECT_ROOT/data"
