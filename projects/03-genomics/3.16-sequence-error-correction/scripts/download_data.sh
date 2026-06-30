#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL reads (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.16 : Sequence Error Correction
#
# CONTRACT (CLAUDE.md sec 8): idempotent, documented, prints source URLs and
# NEVER bypasses credentials/registration. The demo runs on a committed
# SYNTHETIC sample; real benchmark reads are large public archives, so this
# script prints the recipe and defers to make_synthetic.py for an offline
# stand-in. It downloads nothing by itself.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.16 -- Sequence Error Correction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The demo runs on the committed SYNTHETIC sample (data/sample/reads_sample.txt)."
echo "To experiment on REAL reads with known error profiles, use a public archive:"
echo
echo "  GAGE benchmark short reads (known reference -> measurable error rates):"
echo "    http://gage.cbcb.umd.edu/"
echo "  GIAB / Genome in a Bottle (NIST truth sets to score corrected reads):"
echo "    https://www.nist.gov/programs-projects/genome-bottle"
echo "  SRA (ONT / PacBio CLR high-error long reads -- a different regime):"
echo "    https://www.ncbi.nlm.nih.gov/sra"
echo
echo "Convert a FASTA/FASTQ slice into this project's simple text format"
echo "(see data/README.md): one '<n> <has_truth>' header line, then the reads."
echo "Set has_truth=0 when you have no error-free reference."
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py                 # the committed 120-read sample"
echo "  python scripts/make_synthetic.py --reads 200000  # a larger, GPU-friendlier set"
echo
echo "Note: real corrected-read evaluation respects each archive's terms; this"
echo "script never bypasses registration or logins."
