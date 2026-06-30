#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.2 : Short-Read Mapping / Alignment
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. The catalog's short-read datasets are
# either huge (1000 Genomes, SRA) or benchmark sets best fetched with their own
# tooling (GiaB, ENCODE), so this script PRINTS INSTRUCTIONS rather than blindly
# downloading gigabytes. The committed synthetic sample already runs the demo
# offline; scripts/make_synthetic.py makes a larger synthetic problem on demand.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.2 -- Short-Read Mapping / Alignment"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This teaching project ships a tiny SYNTHETIC sample (data/sample/reads_sample.txt)"
echo "that runs the demo offline. Real short-read datasets are large and/or gated:"
echo
echo "  * 1000 Genomes Project (open, very large FASTQ/CRAM):"
echo "      https://www.internationalgenome.org/data"
echo "  * Genome in a Bottle (GiaB) NA12878 / HG002 benchmark WGS + truth sets:"
echo "      https://www.nist.gov/programs-projects/genome-bottle"
echo "      ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/"
echo "  * SRA FASTQ archives (use the SRA Toolkit 'prefetch'/'fasterq-dump'):"
echo "      https://www.ncbi.nlm.nih.gov/sra"
echo "  * ENCODE ChIP/RNA-seq FASTQs (curated functional genomics):"
echo "      https://www.encodeproject.org/"
echo
echo "To run this program on real data, prepare a plain-text file:"
echo "    line 1            = a (short) reference sequence, ACGT only"
echo "    each later line   = one read, ACGT only, all reads the same length"
echo "  then:  ./short-read-mapping-alignment <that-file>"
echo
echo "For a larger SYNTHETIC stand-in (no download, fully offline):"
echo "    python scripts/make_synthetic.py --ref-len 4000 --n-reads 2000"
echo
echo "[download_data] No bytes downloaded (by design). See data/README.md."
