#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get a REAL reference genome (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 3.17 : CRISPR Guide Design & Off-Target Scoring
#
# Prints the recipe for fetching a reference genome (FASTA) to scan against and
# the benchmark guide sets. Downloads nothing and needs no credentials; defers
# to scripts/make_synthetic.py for an offline stand-in (CLAUDE.md §8).
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.17 -- CRISPR Guide Design & Off-Target Scoring"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real off-target scanning runs a guide against a reference genome:"
echo "  1) Download a reference genome (FASTA) from UCSC, e.g. one human"
echo "     chromosome (smaller than the whole genome):"
echo "       https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz"
echo "     (whole genome: https://genome.ucsc.edu/  | mouse: mm10)"
echo "  2) gunzip it, strip the '>' header and newlines to get one ACGT string:"
echo "       zcat chr21.fa.gz | grep -v '^>' | tr -d '\\n' > chr21.seq"
echo "  3) Write the loader format documented in data/README.md:"
echo "       guide  myGuide  <20-letter ACGT spacer>"
echo "       genome <the ACGT string>"
echo "  4) Validated guide efficiencies/off-targets for benchmarking:"
echo "       CRISPOR  https://crispor.gi.ucsc.edu/"
echo "       GeCKO v2 https://www.addgene.org/pooled-library/"
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py --filler 100000   # a larger synthetic genome"
echo
echo "Note: the guide must be exactly 20 ACGT bases; the genome must be at"
echo "least 23 bases (one window). See data/README.md for the exact format."
