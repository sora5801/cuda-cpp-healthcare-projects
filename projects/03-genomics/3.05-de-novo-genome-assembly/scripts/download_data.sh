#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.5 : De Novo Genome Assembly  (all-vs-all read-overlap stage)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs +
# guidance, and NEVER bypasses credentials/registration. Real assembly data is
# gigabytes, so this script GUIDES you to the sources rather than auto-fetching;
# the committed tiny synthetic sample suffices for the demo, and
# scripts/make_synthetic.py scales it up offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.5 -- De Novo Genome Assembly"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project ships a tiny SYNTHETIC sample (data/sample/reads_sample.fasta)"
echo "that is sufficient for the demo. Real de-novo assembly benchmark data is large;"
echo "fetch it from the sources below (none require credentials, but all are GBs):"
echo
echo "  * CHM13 (T2T) human reference   : https://github.com/marbl/CHM13"
echo "  * GenomeArk (vertebrate genomes): https://genomeark.github.io/"
echo "  * Human Pangenome (HPRC)        : https://humanpangenome.org/"
echo "  * SRA PacBio HiFi / ONT reads   : https://www.ncbi.nlm.nih.gov/sra"
echo "      (use the SRA Toolkit: 'prefetch <ACC>' then 'fasterq-dump <ACC>')"
echo
echo "  Convert downloaded reads to the FASTA this demo expects (>header / sequence),"
echo "  or generate a larger SYNTHETIC set offline:"
echo "    python scripts/make_synthetic.py --genome-len 50000 --read-len 1000 --step 200 --error-rate 0.02"
echo
echo "  Idempotent pattern to follow when wiring a real fetch:"
echo "    1) skip the download if the file already exists with the right SHA256"
echo "    2) print source URL + expected size + checksum before downloading"
echo "    3) for any credentialed mirror, print registration instructions ONLY"
