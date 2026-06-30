#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real BQSR-input pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.25 : Base Quality Score Recalibration (BQSR)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real BQSR needs (a) an aligned BAM and (b)
# known-variant VCFs; the committed synthetic sample stands in so the demo runs
# offline. This script only prints where to get the real inputs.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.25 -- Base Quality Score Recalibration (BQSR)"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real BQSR consumes an aligned BAM plus KNOWN-VARIANT VCFs for masking."
echo "This teaching project uses a small SYNTHETIC text alignment instead; to"
echo "experiment with real inputs, fetch these (most are open, no account):"
echo
echo "  dbSNP build 155 (known SNPs) : https://www.ncbi.nlm.nih.gov/snp/"
echo "  Mills & 1000G indels (GATK)  : https://storage.googleapis.com/genomics-public-data/"
echo "  GIAB known-variant VCFs      : https://www.nist.gov/programs-projects/genome-bottle"
echo "  1000 Genomes high-cov WGS    : https://www.internationalgenome.org/data"
echo
echo "Then convert a region to this project's text format (data/README.md):"
echo "  REF/KNOWN/READS lines; one read per line as 'pos bases q0..q(L-1)'."
echo
echo "No download needed for the demo. Bigger SYNTHETIC set:"
echo "  python scripts/make_synthetic.py --reads 50000"
echo
echo "Idempotent pattern when wiring a real fetch:"
echo "  1) skip if the file already exists with the expected SHA256"
echo "  2) print source URL + expected size + checksum"
echo "  3) for credentialed sets, print registration instructions ONLY"
