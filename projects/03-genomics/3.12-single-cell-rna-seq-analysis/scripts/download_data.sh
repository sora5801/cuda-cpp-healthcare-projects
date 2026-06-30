#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.12 : Single-Cell RNA-seq Analysis  (reduced-scope teaching version)
#
# CONTRACT (CLAUDE.md SS8): idempotent, documented, prints the source URLs and
# NEVER bypasses credentials/registration. Real scRNA-seq matrices ship as
# .h5ad / .mtx / 10x HDF5 and need Scanpy/AnnData to export a dense slice into
# THIS project's plain-text format, so this script prints the path rather than
# silently converting. The committed synthetic sample already runs the demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.12 -- Single-Cell RNA-seq Analysis"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "The committed tiny SYNTHETIC sample (data/sample/scrna_sample.txt) is"
echo "enough to build and run the demo offline. No download is required."
echo
echo "For a larger SYNTHETIC problem (still offline, deterministic):"
echo "    python scripts/make_synthetic.py --cells 300 --genes 48 --k 10"
echo
echo "REAL scRNA-seq datasets (from the catalog):"
echo "  * Human Cell Atlas  : https://www.humancellatlas.org/"
echo "  * 10x Genomics sets : https://www.10xgenomics.com/resources/datasets"
echo "  * CellxGene Census  : https://cellxgene.cziscience.com/   (50M+ cells)"
echo "  * NCBI GEO          : https://www.ncbi.nlm.nih.gov/geo/"
echo
echo "To use a real matrix here, export a dense slice with Scanpy in Python:"
echo "    import scanpy as sc"
echo "    a = sc.read_10x_h5('filtered_feature_bc_matrix.h5')   # or sc.read_h5ad(...)"
echo "    a = a[:300, :48]                                      # tiny teaching slice"
echo "    # then write 'N G k target_sum' + rows of '<label> count0..' as in data/README.md"
echo
echo "Some assets require (free) registration. This script will NOT bypass any"
echo "login or license; it only points you to the source pages above."
