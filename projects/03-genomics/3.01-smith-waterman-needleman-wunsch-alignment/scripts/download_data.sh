#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL sequences (Linux/macOS)
# ---------------------------------------------------------------------------
# Project 3.01 : Smith-Waterman / Needleman-Wunsch Alignment
# Prints how to obtain two FASTA sequences; downloads nothing. See section 8.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 3.01 -- Smith-Waterman / Needleman-Wunsch Alignment"
echo
echo "Pick two sequences from a public FASTA database, e.g.:"
echo "  UniProt/Swiss-Prot : https://www.uniprot.org/downloads"
echo "  NCBI RefSeq        : https://ftp.ncbi.nlm.nih.gov/refseq/"
echo
echo "Write them as two lines (this project uses a DNA A/C/G/T alphabet):"
echo "  line 1 = query sequence ; line 2 = target sequence"
echo "Save as data/sample/sequences_sample.txt (strip FASTA '>' headers/newlines)."
echo
echo "Offline stand-in (no download, reproducible):"
echo "  python scripts/make_synthetic.py --motif 400 --mut 0.2"
echo
echo "Target data dir: $PROJECT_ROOT/data"
