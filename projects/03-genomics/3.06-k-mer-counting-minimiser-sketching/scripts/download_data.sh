#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.6 -- k-mer Counting & Minimiser Sketching
#
# CONTRACT (CLAUDE.md section 8): idempotent, documented, prints source URL +
# expected size, and NEVER bypasses credentials/registration. The real read sets
# (NA12878 WGS, GenomeTrakr, GAGE) live in NCBI's Sequence Read Archive and are
# fetched with the SRA Toolkit -- a separate install. This script PRINTS the
# exact reproducible commands and defers to scripts/make_synthetic.py for the
# offline demo. The committed tiny sample already runs the demo with no download.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
FULL_DIR="$DATA_DIR/full"

echo "[download_data] Project 3.6 -- k-mer Counting & Minimiser Sketching"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# Idempotence: if a FASTQ already sits in data/full/, do nothing.
if ls "$FULL_DIR"/*.fastq* >/dev/null 2>&1; then
  echo "[download_data] data/full already contains FASTQ files -- nothing to do."
  exit 0
fi

cat <<'EOF'
The real read sets are public but large; fetch them with the NCBI SRA Toolkit:

  1) Install the SRA Toolkit:  https://github.com/ncbi/sra-tools  (prefetch + fasterq-dump)
  2) NA12878 Illumina WGS (human reference; accession SRR622457, tens of GB):
       prefetch SRR622457
       fasterq-dump --split-files -O data/full SRR622457
  3) GenomeTrakr pathogen WGS (bacterial surveillance):  BioProject PRJNA183844
       https://www.ncbi.nlm.nih.gov/bioproject/PRJNA183844
  4) GAGE assembly benchmark (multi-species short reads):
       http://gage.cbcb.umd.edu/
  5) SRA front door (find more accessions):  https://www.ncbi.nlm.nih.gov/sra

These are open-access; no credentials are required, but the SRA Toolkit is a
separate install -- we never bundle or bypass it.

For an offline OR larger SYNTHETIC problem, use:
    python scripts/make_synthetic.py --genome 200000 --reads 5000 --readlen 100
EOF
