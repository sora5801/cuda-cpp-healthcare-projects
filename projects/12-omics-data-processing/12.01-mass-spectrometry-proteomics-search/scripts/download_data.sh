#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Real proteomics-data pointers (Linux/macOS)
# Project 12.01 : Mass-Spectrometry Proteomics Search. Nothing to download.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 12.01 -- Mass-Spectrometry Proteomics Search"
echo
echo "Real data: observed MS/MS spectra (mzML) searched against a peptide DB."
echo "Bin observed peaks + theoretical fragments to a common grid, then write the"
echo "format in data/README.md."
echo
echo "  ProteomeXchange / PRIDE : https://www.proteomexchange.org  (raw/mzML)"
echo "  MSFragger               : https://github.com/Nesvilab/MSFragger"
echo "  GiCOPS (GPU search)     : https://github.com/pcdslab/gicops"
echo "  OpenMS                  : https://github.com/OpenMS/OpenMS  (mzML I/O)"
echo
echo "Bigger synthetic set (no download):"
echo "  python scripts/make_synthetic.py --N 8192"
echo
echo "Target data dir: $PROJECT_ROOT/data"
