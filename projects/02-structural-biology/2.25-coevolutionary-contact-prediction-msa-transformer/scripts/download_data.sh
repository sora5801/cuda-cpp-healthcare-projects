#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real coevolution-MSA pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 2.25 : Coevolutionary Contact Prediction & MSA Transformer
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real coevolution needs a DEEP MSA of a real
# protein family; building or downloading one is a multi-GB, tool-heavy step, so
# this script only prints the pointers and defers to the committed synthetic
# sample (or scripts/make_synthetic.py) for an offline, runnable demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 2.25 -- Coevolutionary Contact Prediction & MSA Transformer"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Nothing is auto-downloaded. The program reads an aligned-FASTA MSA"
echo "(one record per sequence, all the same length). To use a REAL family:"
echo
echo "  Pfam family MSAs   : http://pfam.xfam.org           (Stockholm -> aligned FASTA)"
echo "  UniRef50/UniRef90  : https://www.uniprot.org/help/uniref  (build an MSA via jackhmmer/HHblits)"
echo "  EVcouplings        : https://github.com/debbiemarkslab/EVcouplings  (benchmark families + PDB contacts)"
echo "  CASP14 contacts    : https://predictioncenter.org   (community contact benchmark)"
echo "  ESM-MSA-1b         : https://github.com/facebookresearch/esm  (MSA Transformer, the deep-learning route)"
echo
echo "Build an MSA, save it as aligned FASTA, then run:"
echo "  ./build/cmake/coevolutionary-contact-prediction-msa-transformer path/to/family.fasta"
echo
echo "No download needed for the demo -- the committed synthetic sample suffices."
echo "Bigger synthetic MSA (deeper, sharper signal):"
echo "  python scripts/make_synthetic.py --n 4000 --seed 7"
