#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real protein-LM data pointers (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 3.18 : Protein Language Model Inference. Nothing to download: the demo
# generates all model weights deterministically (src/attention_math.h) and ships
# a synthetic sample sequence. This script only prints where the REAL trained
# models and sequence corpora live (CLAUDE.md §8: never bypass credentials).
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 3.18 -- Protein Language Model Inference"
echo
echo "This teaching demo needs NO download: it generates synthetic weights in"
echo "code and ships a synthetic sample sequence in data/sample/."
echo
echo "For a REAL protein language model:"
echo "  Trained models (ESM-2 / ESMFold) : https://github.com/facebookresearch/esm"
echo "  EvolutionaryScale ESM3           : https://github.com/evolutionaryscale/esm"
echo "  Sequence corpus (UniRef50/90)    : https://www.uniprot.org/help/uniref"
echo "  ESM Metagenomic Atlas            : https://esmatlas.com/"
echo "  Structural validation (PDB)      : https://www.rcsb.org/"
echo "  CATH / SCOP classification       : https://www.cathdb.info/"
echo
echo "ESM-2 weights are large (hundreds of MB to tens of GB) and are NOT"
echo "redistributed here; fetch them via fair-esm's torch.hub / transformers APIs."
echo
echo "Longer synthetic peptide (no download):"
echo "  python scripts/make_synthetic.py --len 64"
echo
echo "Target data dir: $PROJECT_ROOT/data"
