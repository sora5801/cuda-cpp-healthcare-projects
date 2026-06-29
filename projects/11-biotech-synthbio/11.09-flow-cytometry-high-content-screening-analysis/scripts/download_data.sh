#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real cytometry-data pointers (Linux/macOS)
# Project 11.09 : Flow Cytometry & High-Content Screening Analysis. Nothing to fetch.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 11.09 -- Flow Cytometry & High-Content Screening Analysis"
echo
echo "Real data is in FCS files; export a few markers per event into the"
echo "format in data/README.md ('N D K' then N rows of D floats in [0,1])."
echo
echo "  FlowRepository : http://flowrepository.org      (public FCS datasets)"
echo "  FlowKit        : https://github.com/whitews/FlowKit   (read/transform FCS)"
echo "  RAPIDS cuML    : https://github.com/rapidsai/cuml     (GPU clustering)"
echo
echo "Bigger synthetic set (no download):"
echo "  python scripts/make_synthetic.py --scale 50"
echo
echo "Target data dir: $PROJECT_ROOT/data"
