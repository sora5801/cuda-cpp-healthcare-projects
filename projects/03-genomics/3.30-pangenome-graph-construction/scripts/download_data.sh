#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch / build the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.30 : Pangenome Graph Construction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real pangenome graphs are BUILT from genome
# assemblies with the PGGB pipeline (not a single download), so this script prints
# the exact, reproducible recipe and links rather than fetching opaque blobs. The
# committed synthetic sample is enough to run the demo offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.30 -- Pangenome Graph Construction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project lays out a pangenome GRAPH; real graphs are built"
echo "from genome assemblies. There is no single file to download -- you build"
echo "the graph with PGGB, then convert its GFA into this project's format."
echo
echo "Assembly sources (respect each license; some require registration):"
echo "  * HPRC year-1 (94 human haplotypes) : https://humanpangenome.org/"
echo "  * Ensembl (non-human pangenomes)    : https://www.ensembl.org/"
echo "  * Vertebrate Genomes Project        : https://vertebrategenomesproject.org/"
echo "  * NCBI RefSeq (bacterial pangenomes): https://ftp.ncbi.nlm.nih.gov/refseq/"
echo
echo "Reproducible recipe:"
echo "  1) Put your assemblies into one FASTA:  cat *.fa > seqs.fa; samtools faidx seqs.fa"
echo "  2) Build the graph (Docker):"
echo "       docker run -v \$PWD:/data ghcr.io/pangenome/pggb:latest \\"
echo "         pggb -i /data/seqs.fa -o /data/out -n <num_haplotypes> -t 8 -p 90 -s 5000"
echo "  3) The graph is out/*.gfa . Convert its S (segments) and P/W (paths) lines"
echo "     into this project's 'N P / lengths / paths' format (see data/README.md)."
echo
echo "For a larger SYNTHETIC graph (no download, fully offline):"
echo "  python scripts/make_synthetic.py        # writes data/sample/pangenome_sample.txt"
echo
echo "[download_data] Nothing fetched. The committed synthetic sample runs the demo."
