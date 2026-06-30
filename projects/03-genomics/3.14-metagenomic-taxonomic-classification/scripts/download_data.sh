#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL reference data (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.14 : Metagenomic Taxonomic Classification
#
# This project's "real data" is a reference k-mer database built from genomes
# plus a set of sequencing reads. There is no single file to fetch; this script
# prints the recipe and defers to make_synthetic.py for an offline stand-in
# (CLAUDE.md section 8). It requires no credentials and downloads nothing itself.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 3.14 -- Metagenomic Taxonomic Classification"
echo
echo "The committed sample in data/sample/metagenome_sample.txt is SYNTHETIC and"
echo "is all you need to build and run the demo offline."
echo
echo "To classify REAL reads against REAL genomes, build a reference database:"
echo "  1) Reference genomes (open access):"
echo "       NCBI RefSeq microbial genomes : https://ftp.ncbi.nlm.nih.gov/refseq/"
echo "       (a prebuilt Kraken2 standard DB is the production form of our hash map)"
echo "  2) Benchmark metagenomes with known truth (for accuracy):"
echo "       CAMI challenge datasets       : https://data.cami-challenge.org/"
echo "  3) Real-world reads:"
echo "       Human Microbiome Project      : https://www.hmpdacc.org/"
echo "       SRA metagenomics projects     : https://www.ncbi.nlm.nih.gov/sra"
echo "       (HMP/SRA may require registration -- follow their instructions; this"
echo "        script will NOT bypass any access control.)"
echo
echo "  4) Convert to this project's format (data/README.md):"
echo "       - one REF line per taxon (taxon_name + reference sequence)"
echo "       - one READ line per read (true taxon id or 0, + read sequence)"
echo "     then run the demo on your file:"
echo "       ./build/cmake/metagenomic-taxonomic-classification your_data.txt"
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py --reads 100000    # a larger synthetic run"
echo
echo "Tip: keep KMER_K (=15) consistent with src/kmer_core.h."
echo "Target data dir: $PROJECT_ROOT/data"
