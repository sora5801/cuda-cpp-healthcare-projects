#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  How to get REAL CT data (Linux/macOS)
# Project 4.01 : CT Reconstruction (Filtered Backprojection). Downloads nothing.
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[download_data] Project 4.01 -- CT Reconstruction (Filtered Backprojection)"
echo
echo "Options for real/standard data:"
echo "  * Shepp-Logan digital phantom: generate via ASTRA/TIGRE, forward-project,"
echo "    write the sinogram in data/README.md format."
echo "  * TCIA (https://www.cancerimagingarchive.net): real DICOM CT data (may need"
echo "    registration; this script does NOT bypass it)."
echo "  * Toolkits with sample data: RTK, ASTRA, TIGRE, Plastimatch."
echo
echo "Offline stand-in (no download, reproducible):"
echo "  python scripts/make_synthetic.py --angles 360 --det 367 --img 256"
echo
echo "Target data dir: $PROJECT_ROOT/data"
