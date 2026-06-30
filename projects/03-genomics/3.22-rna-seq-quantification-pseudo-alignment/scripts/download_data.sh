#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Point at the FULL datasets (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.22 : RNA-seq Quantification / Pseudo-alignment
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Real pseudo-alignment needs (a) a reference
# transcriptome FASTA and (b) RNA-seq FASTQs, then a tool (kallisto / Salmon) to
# PRODUCE the equivalence classes this project consumes. That pipeline is outside
# the scope of a single teaching demo, so this script only prints the canonical
# sources + the exact commands to reproduce ec counts, and otherwise defers to
# scripts/make_synthetic.py for an offline, fully-reproducible stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.22 -- RNA-seq Quantification / Pseudo-alignment"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project consumes EQUIVALENCE CLASSES (ec counts), which are produced"
echo "by running a pseudo-aligner on real reads. The canonical inputs are:"
echo
echo "  Reference transcriptome (FASTA):"
echo "    GENCODE human transcriptome   https://www.gencodegenes.org/"
echo
echo "  RNA-seq reads (FASTQ):"
echo "    ENCODE RNA-seq               https://www.encodeproject.org/"
echo "    GTEx v9 tissue compendium    https://gtexportal.org/   (registration)"
echo "    SRA RNA-seq studies          https://www.ncbi.nlm.nih.gov/sra"
echo
echo "  To PRODUCE ecs from those (kallisto's output includes them):"
echo "    kallisto index -i idx gencode.transcripts.fa.gz"
echo "    kallisto quant -i idx -o out --plaintext reads_1.fastq.gz reads_2.fastq.gz"
echo "    # out/ then holds run_info.json + the ec / abundance tables to reformat"
echo "    # into this project's 'T M / eff lengths / ec lines / TRUTH' text layout."
echo
echo "GTEx and some SRA studies require registration/credentials -- this script"
echo "does NOT attempt to bypass that. For an offline, reproducible run, use the"
echo "committed synthetic sample (already in data/sample/) or regenerate it:"
echo "    python scripts/make_synthetic.py --reads 1000000"
