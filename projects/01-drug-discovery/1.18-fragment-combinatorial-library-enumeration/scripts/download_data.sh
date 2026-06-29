#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL synthon descriptors (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 1.18 : Fragment / Combinatorial Library Enumeration
#
# This project's "real data" is DESCRIPTORS computed from building-block SMILES
# with RDKit, not a single downloadable file. Building-block catalogs require
# registration, so this script does NOT bypass credentials (CLAUDE.md sec.8): it
# prints the recipe and defers to scripts/make_synthetic.py for an offline,
# reproducible stand-in. It downloads nothing and is safe to re-run (idempotent).
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 1.18 -- Fragment / Combinatorial Library Enumeration"
echo
echo "Real synthon descriptors are computed from building-block SMILES with RDKit:"
echo "  1) Obtain a building-block catalog (registration required):"
echo "       Enamine building blocks : https://enamine.net/building-blocks"
echo "       Enamine REAL Space      : https://enamine.net"
echo "       ChemSpace               : https://chem-space.com"
echo "     Group the blocks by reactive class into 3 reactant slots (e.g. an"
echo "     Ugi-like amine / aldehyde-or-acid / isocyanide-cap scheme)."
echo "  2) pip install rdkit"
echo "  3) For each building block, compute the 5 additive descriptors:"
echo "       MW    = Descriptors.MolWt(mol)"
echo "       cLogP = Crippen.MolLogP(mol)"
echo "       TPSA  = rdMolDescriptors.CalcTPSA(mol)"
echo "       HBD   = rdMolDescriptors.CalcNumHBD(mol)"
echo "       HBA   = rdMolDescriptors.CalcNumHBA(mol)"
echo "  4) Write the catalog text format documented in data/README.md"
echo "     (N_SLOTS, then per slot 'SLOT k size' + one row per building block)."
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py --per-slot 40    # 40^3 = 64000 products"
echo
echo "Tip: keep N_SLOTS (=3) consistent with src/product_core.h."
echo "Target data dir: $PROJECT_ROOT/data"
