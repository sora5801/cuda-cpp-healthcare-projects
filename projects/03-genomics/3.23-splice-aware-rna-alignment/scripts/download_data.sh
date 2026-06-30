#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Pointers to REAL RNA-seq data (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.23 : Splice-Aware RNA Alignment   (REDUCED-SCOPE teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and
# NEVER bypasses credentials/registration. This project's DEMO needs no download
# -- the committed synthetic sample (data/sample/reads_sample.txt) is enough.
# This script just guides you to real data if you want to go further.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.23 -- Splice-Aware RNA Alignment"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project runs fully on the committed SYNTHETIC sample:"
echo "    data/sample/reads_sample.txt   (a 3-exon gene + 6 reads)"
echo "No download is required for the demo."
echo
echo "To experiment with REAL RNA-seq, fetch from these sources yourself:"
echo
echo "  * ENCODE RNA-seq FASTQs + GENCODE annotation (open access):"
echo "      https://www.encodeproject.org/    (reads)"
echo "      https://www.gencodegenes.org/     (GTF gene models = true exons/introns)"
echo "  * SRA RNA-seq benchmarks (SEQC/MAQC):"
echo "      https://www.ncbi.nlm.nih.gov/sra"
echo "      (use the SRA Toolkit 'prefetch' + 'fasterq-dump' to get FASTQ)"
echo "  * GTEx tissue RNA-seq (CONTROLLED ACCESS -- individual-level via dbGaP):"
echo "      https://gtexportal.org/"
echo "      Register at dbGaP; this script does NOT and CANNOT bypass that."
echo
echo "Recommended idempotent pattern when you wire a fetch in:"
echo "    1) skip the download if the file already exists with the right SHA256"
echo "    2) print the source URL + expected size + SHA256 before downloading"
echo "    3) for controlled-access sets, print registration instructions ONLY"
echo
echo "For a larger SYNTHETIC batch instead, just re-seed the generator:"
echo "    python scripts/make_synthetic.py --seed 7"
