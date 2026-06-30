#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.28 : Profile HMM (Viterbi / Forward)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. The committed tiny SYNTHETIC sample in
# data/sample/ already lets the demo run offline -- this script only points you at
# the real corpora and shows how to turn them into the loader's FASTA format.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.28 -- Profile HMM (Viterbi / Forward)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project ships a tiny SYNTHETIC sample (data/sample/phmm_sample.fasta)"
echo "that is sufficient to build, run, and verify the demo offline. The real-world"
echo "corpora below are large and/or governed by their own licenses:"
echo
echo "  Pfam-A (profile HMMs for ~20k protein families)"
echo "    https://www.ebi.ac.uk/interpro/download/   (Pfam-A.hmm.gz)"
echo "    License: CC0. Use HMMER's 'hmmemit -c' to extract a family CONSENSUS,"
echo "    then place it as record 0 of a FASTA file (see data/README.md)."
echo
echo "  UniRef50 (clustered protein sequences to search)"
echo "    https://www.uniprot.org/help/uniref       (uniref50.fasta.gz)"
echo "    License: CC BY 4.0. These become the DATABASE records (>=1 per sequence)."
echo
echo "  Rfam (RNA family profiles)   https://rfam.org/"
echo "  JGI metagenome proteins      https://genome.jgi.doe.gov/  (registration required)"
echo
echo "  This project's loader expects a simple FASTA-like file:"
echo "     >name <newline> AMINOACIDS <newline> ...   (record 0 = consensus)."
echo "  Only the 20 standard amino acids are supported, and MAX_M=64 / MAX_L=256"
echo "  (see src/phmm.h). Trim longer Pfam profiles or raise the caps + rebuild."
echo
echo "  For a larger SYNTHETIC stand-in (more decoys), run:"
echo "     python scripts/make_synthetic.py --decoys 64"
