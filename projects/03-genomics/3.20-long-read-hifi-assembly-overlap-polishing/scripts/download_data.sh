#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.20 : Long-Read HiFi Assembly Overlap & Polishing
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs +
# expected sizes, and NEVER bypasses credentials/registration. The committed
# synthetic sample already runs the demo offline, so a download is OPTIONAL.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.20 -- Long-Read HiFi Assembly Overlap & Polishing"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed tiny sample in data/sample/reads_sample.txt is enough to run"
echo "the demo offline. No download is required for the teaching demo."
echo
echo "FULL public PacBio HiFi datasets (no credentials needed, but large):"
echo "  * Human Pangenome / HG002 HiFi reads (SRA):"
echo "      https://www.ncbi.nlm.nih.gov/sra  (search 'HG002 PacBio HiFi')"
echo "  * Vertebrate Genomes Project HiFi assemblies:"
echo "      https://vertebrategenomesproject.org/"
echo "  * GenomeArk HiFi datasets (AWS open data):"
echo "      https://genomeark.github.io/"
echo "  * CHM13 T2T HiFi reads:"
echo "      https://github.com/marbl/CHM13"
echo
echo "To turn raw HiFi reads (FASTQ/BAM) into the minimiser-sketch format this"
echo "project consumes, the standard tools are:"
echo "  * minimap2  (k/w minimiser sketching + overlap):  https://github.com/lh3/minimap2"
echo "  * hifiasm   (state-of-the-art HiFi assembler):     https://github.com/chhylp123/hifiasm"
echo
echo "For a larger SYNTHETIC overlap graph that runs WITHOUT any download:"
echo "  python scripts/make_synthetic.py --n-reads 2000"
echo
echo "When wiring a real dataset, keep the fetch idempotent: skip the download if"
echo "the file already exists with the right size/checksum, and print the source"
echo "URL + expected size before downloading."
