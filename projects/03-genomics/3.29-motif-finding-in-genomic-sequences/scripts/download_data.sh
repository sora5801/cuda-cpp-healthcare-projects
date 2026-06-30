#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL ChIP-seq motif input (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 3.29 : Motif Finding in Genomic Sequences
#
# Real input is a multi-FASTA of ChIP-seq peak sequences, assembled from a peak
# BED + a reference genome (no single file to grab). Prints the recipe + tool
# links; downloads nothing and needs no credentials. Use make_synthetic.py for an
# offline stand-in (CLAUDE.md sec 8).
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.29 -- Motif Finding in Genomic Sequences"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real input is a FASTA of transcription-factor ChIP-seq peak sequences:"
echo "  1) Pick a TF ChIP-seq experiment and download its peak BED, e.g."
echo "       ENCODE  : https://www.encodeproject.org/   (thousands of TF experiments)"
echo "       ReMap   : https://remap.univ-amu.fr/        (~5k experiments)"
echo "       GEO     : https://www.ncbi.nlm.nih.gov/geo/"
echo "  2) Extract peak summit +/- ~100 bp from a reference genome FASTA:"
echo "       bedtools getfasta -fi genome.fa -bed peaks.bed -fo peaks.fasta"
echo "  3) Feed peaks.fasta to the program (or to MEME / HOMER for comparison)."
echo "  4) Validate the recovered motif against a known PWM in JASPAR 2024:"
echo "       https://jaspar.elixir.no/"
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py --n 5000 --len 200"
echo
echo "Note: licenses vary by dataset -- respect each source's terms; do not"
echo "redistribute genome-derived sequence unless the license permits it."
