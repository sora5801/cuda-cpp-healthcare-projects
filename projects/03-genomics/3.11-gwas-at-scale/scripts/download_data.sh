#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.11 : GWAS at Scale
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URL +
# access terms, and NEVER bypasses credentials/registration. Real GWAS cohorts
# are controlled-access and cannot be redistributed, so this script does NOT
# download genotypes -- it prints exactly how to obtain them legally and defers
# to scripts/make_synthetic.py for an offline stand-in.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.11 -- GWAS at Scale"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Real GWAS cohorts are CONTROLLED-ACCESS and may NOT be redistributed."
echo "This script will not (and cannot) bypass that. To use real data:"
echo
echo "  UK Biobank  (~500k individuals, ~800k variants)"
echo "    Apply for access: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access"
echo "    Genotypes ship as PLINK .bed/.bim/.fam or BGEN; convert with PLINK2."
echo
echo "  dbGaP  (controlled-access GWAS datasets)"
echo "    Data Access Request: https://www.ncbi.nlm.nih.gov/gap/"
echo
echo "  GWAS Catalog  (OPEN published summary statistics -- no genotypes)"
echo "    Browse / download: https://www.ebi.ac.uk/gwas/  (useful to cross-check hits)"
echo
echo "  gnomAD  (OPEN allele-frequency / LD reference panels)"
echo "    https://gnomad.broadinstitute.org/"
echo
echo "A real loader would read PLINK2 .bed/.pgen or BGEN, not this demo's text format."
echo
echo "The committed tiny SYNTHETIC sample (data/sample/gwas_sample.txt) already lets"
echo "the demo run offline. For a larger synthetic cohort, run:"
echo "    python scripts/make_synthetic.py --n 2000 --m 5000 --out data/sample/gwas_big.txt"
