#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.27 -- Suffix Array / BWT / FM-Index Construction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size, and NEVER bypasses credentials/registration. The demo runs on the
# committed synthetic sample; the real genomes below are optional. This script
# prints instructions + links and defers to scripts/make_synthetic.py for an
# offline stand-in -- it does not auto-download multi-gigabyte genomes.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.27 -- Suffix Array / BWT / FM-Index Construction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed sample (data/sample/dna_sample.txt) is SYNTHETIC and is all"
echo "the demo needs. The datasets below are the real-world targets; building a"
echo "BWT over a 3 Gb genome requires the external-memory methods described in"
echo "THEORY.md section 7 and is beyond this in-core teaching version."
echo
echo "Real datasets (open access -- no credentials required):"
echo "  * GRCh38 human reference (~3.1 Gb):"
echo "      https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.40/"
echo "  * 1000 Genomes read collections (pan-read BWT):"
echo "      https://www.internationalgenome.org/data"
echo "  * NCBI RefSeq complete microbial genomes (small, good for scaling):"
echo "      https://ftp.ncbi.nlm.nih.gov/refseq/"
echo "  * Human Pangenome sequences (terabase-scale frontier):"
echo "      https://humanpangenome.org/"
echo
echo "To create a larger SYNTHETIC problem that this build CAN handle in-core:"
echo "    python scripts/make_synthetic.py --n 100000"
echo
echo "If you wire a real FASTA fetch here, keep it idempotent:"
echo "    1) skip the download if the file already exists with the right checksum"
echo "    2) strip FASTA headers/newlines to a single A/C/G/T line before use"
echo "    3) print the source URL + expected size + SHA256"
