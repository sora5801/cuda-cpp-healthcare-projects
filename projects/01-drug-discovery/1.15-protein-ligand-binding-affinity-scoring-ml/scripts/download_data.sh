#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 1.15 : Protein-Ligand Binding Affinity Scoring (ML)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL +
# registration steps, and NEVER bypasses credentials. PDBbind / CASF require a
# (free) account and have redistribution terms, so this script prints
# instructions only and defers to scripts/make_synthetic.py for the offline demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 1.15 -- Protein-Ligand Binding Affinity Scoring (ML)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample in data/sample/complexes_sample.txt is SYNTHETIC and is"
echo "enough to build, run, and verify the demo offline. The real benchmarks below"
echo "require (free) registration and are NOT auto-downloaded:"
echo
echo "  PDBbind v2020  -- 19,443 complexes with measured Kd/Ki (training set)"
echo "                    http://www.pdbbind.org.cn   (register, then download)"
echo "  CASF-2016      -- scoring/ranking/docking benchmark"
echo "                    http://www.pdbbind.org.cn/casf.php"
echo "  ChEMBL         -- bioactivity database"
echo "                    https://www.ebi.ac.uk/chembl/"
echo "  BindingDB      -- 2.8M measured binding affinities"
echo "                    https://www.bindingdb.org"
echo
echo "To convert a real complex into this project's input format:"
echo "  1) parse the protein .pdb and ligand .sdf/.mol2 (e.g. with RDKit / Biopython)"
echo "  2) center a 16 A box on the binding pocket; keep atoms with element in {C,N,O,S}"
echo "  3) emit one line per complex: '<m> <pKd>' then m lines '<x> <y> <z> <type> <is_ligand>'"
echo "     (type: 0=C 1=N 2=O 3=S; is_ligand: 0=protein 1=ligand; coords in A in [0,16))"
echo
echo "For a larger SYNTHETIC batch to stress the GPU path instead, run:"
echo "  python scripts/make_synthetic.py --n 100000"
