#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real methylation-data pointers (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 3.24 : Methylation / Modified-Base Calling
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. This project SHIPS SYNTHETIC DATA
# (data/sample/) and needs no download to run the demo; this script only points
# at real datasets for further study and defers to make_synthetic.py for scale.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 3.24 -- Methylation / Modified-Base Calling"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "Nothing to download: the committed synthetic sample in data/sample/ runs the demo."
echo
echo "Real data for further study (basecall + event-align first; see data/README.md):"
echo "  ONT open datasets (R10.4.1, 5mC/6mA labels) : https://github.com/GoekeLab/awesome-nanopore"
echo "  ENCODE WGBS (ground-truth methylation)      : https://www.encodeproject.org/"
echo "  NCBI GEO methylation studies                : https://www.ncbi.nlm.nih.gov/geo/"
echo
echo "Tools that produce the per-site event windows this project consumes:"
echo "  f5c    (CUDA event alignment + meth calling) : https://github.com/hasindu2008/f5c"
echo "  Dorado (basecalling + mod calling)           : https://github.com/nanoporetech/dorado"
echo "  Remora (modified-base models)                : https://github.com/nanoporetech/remora"
echo
echo "Bigger synthetic instance (no download):"
echo "  python scripts/make_synthetic.py --sites 4096"
