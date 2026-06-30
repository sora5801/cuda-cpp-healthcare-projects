#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch / point at the FULL datasets (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 2.16 : Delta-Delta-G Stability Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The real Delta-Delta-G study sets are large
# or license-governed, so this script only prints instructions + links and defers
# to scripts/make_synthetic.py for an offline stand-in. The committed tiny sample
# already runs the demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.16 -- Delta-Delta-G Stability Prediction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed synthetic sample (data/sample/protein_sample.txt) is all the"
echo "demo needs. The datasets below are OPTIONAL study material; each has its own"
echo "license -- respect it. This script does NOT download or bypass registration;"
echo "it prints where to get the data."
echo
echo "  1) Protherm / ProThermDB  (>25k experimental Delta-Delta-G values)"
echo "     https://www.abren.net/protherm/   (see ProThermDB for the successor)"
echo
echo "  2) Megascale stability dataset (Rocklin lab, ~2.5M measurements)"
echo "     https://github.com/Rocklin-Lab/cdna-display-proteolysis-datasets"
echo
echo "  3) ProteinGym substitution/indel benchmarks"
echo "     https://github.com/OATML-Markslab/ProteinGym"
echo
echo "  4) S669 curated single-mutation stability benchmark"
echo "     (verify the current canonical URL in the literature)"
echo
echo "To make a LARGER synthetic protein for scaling experiments, run e.g.:"
echo "    python scripts/make_synthetic.py --residues 512 --out data/sample/protein_big.txt"
echo
echo "To turn a real PDB structure into this project's input, compute a per-residue"
echo "burial fraction (relative solvent accessibility via DSSP or freesasa) and emit"
echo "'<AA> <buried>' lines -- see THEORY.md."
