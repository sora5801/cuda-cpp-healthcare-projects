#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.3 : Variant Calling Acceleration
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + size, and
# NEVER bypasses credentials/registration. The real benchmark resources are
# large and/or access-controlled, so this script prints instructions + links
# ONLY and defers to scripts/make_synthetic.py for the offline stand-in the demo
# actually uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.3 -- Variant Calling Acceleration"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a tiny SYNTHETIC sample (data/sample/) so the demo runs"
echo "offline. No real dataset is downloaded automatically -- the real benchmark"
echo "resources are large and some require registration."
echo
echo "REAL BENCHMARK RESOURCES (open in a browser, follow each site's terms):"
echo "  * GiaB truth sets HG001-HG007 (gold-standard germline calls):"
echo "      https://www.nist.gov/programs-projects/genome-bottle"
echo "  * ClinVar (clinically interpreted variants):"
echo "      https://www.ncbi.nlm.nih.gov/clinvar/"
echo "  * gnomAD v4 (population allele frequencies):"
echo "      https://gnomad.broadinstitute.org/"
echo "  * 1000 Genomes high-coverage WGS:"
echo "      https://www.internationalgenome.org/data"
echo
echo "TO USE REAL DATA with this teaching kernel:"
echo "  1) Pick one locus; extract candidate haplotypes (local assembly of the"
echo "     active region) and overlapping reads from a BAM."
echo "  2) Convert to the text format in data/README.md (haplotypes + reads +"
echo "     Phred qualities) and pass the file path as argv[1] to the exe."
echo
echo "FOR A LARGER SYNTHETIC PROBLEM (no download needed):"
echo "    python scripts/make_synthetic.py --reads 4096 --read-len 100 --hap-len 120"
echo
echo "[download_data] Nothing downloaded (by design). The demo uses the synthetic sample."
