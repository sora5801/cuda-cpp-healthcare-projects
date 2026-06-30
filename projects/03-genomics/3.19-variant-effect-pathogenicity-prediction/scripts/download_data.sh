#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.19 -- Variant Effect / Pathogenicity Prediction
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs + guidance,
# and NEVER bypasses credentials/registration. The committed tiny SYNTHETIC
# sample (data/sample/variants_sample.txt) already lets the demo run offline, so
# this script only points you at the real, labelled resources.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.19 -- Variant Effect / Pathogenicity Prediction"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This educational project ships a SYNTHETIC sample and a fixed, UNTRAINED toy"
echo "model, so no download is required to run the demo. To experiment with REAL"
echo "variant-effect resources, fetch them yourself from the sources below and"
echo "respect every license (do NOT commit redistribution-restricted data):"
echo
echo "  ClinVar (public)  pathogenic/benign calls"
echo "    https://www.ncbi.nlm.nih.gov/clinvar/"
echo "    VCF: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
echo "  gnomAD (public)   allele frequencies + per-gene constraint"
echo "    https://gnomad.broadinstitute.org/"
echo "  MaveDB (public)   deep mutational scanning (DMS) atlas"
echo "    https://www.mavedb.org/"
echo "  HGMD (LICENSED -- registration required; NOT auto-downloaded here)"
echo "    http://www.hgmd.cf.ac.uk/"
echo
echo "For a larger SYNTHETIC batch (e.g. to see the GPU overtake the CPU), run:"
echo "    python scripts/make_synthetic.py --n 2000000"
echo
echo "When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
