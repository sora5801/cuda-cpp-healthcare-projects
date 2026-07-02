#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.13 : Gene Regulatory Network Inference (ARACNE: MI + DPI)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. This project's
# "real data" is single-cell RNA-seq for which there is no committable ground-
# truth network, so the demo runs on a labeled-synthetic sample; this script
# prints how to obtain real data and defers to make_synthetic.py offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.13 -- Gene Regulatory Network Inference"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project ships a labeled-SYNTHETIC sample with a KNOWN ground-truth"
echo "network (TF->A->B, TF->C, D->E; F,G,H,I noise) so you can watch ARACNE"
echo "recover it. Real scRNA-seq has no such ground truth to redistribute."
echo
echo "To try REAL data, obtain an expression matrix (genes x cells) from:"
echo "  * Gene Expression Omnibus (GEO)        https://www.ncbi.nlm.nih.gov/geo/"
echo "  * BEELINE benchmark GRN datasets       https://github.com/Murali-group/BEELINE"
echo "  * Human Cell Atlas scRNA-seq           https://www.humancellatlas.org"
echo "  * ENCODE TF binding ChIP-seq (truth)   https://www.encodeproject.org"
echo "then reshape it to this loader's text format (see data/README.md):"
echo "  line 1: '<n_genes> <n_samples>'; then one row per gene: '<name> v0 v1 ...'"
echo
echo "Offline stand-in (no download, fully reproducible):"
echo "  python scripts/make_synthetic.py --samples 400"
echo
echo "Note: BEELINE/GEO/HCA are large and license-bound; respect each license."
echo "This script downloads nothing by itself and never bypasses credentials."
