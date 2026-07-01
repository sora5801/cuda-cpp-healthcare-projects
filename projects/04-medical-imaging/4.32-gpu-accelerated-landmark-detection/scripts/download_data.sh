#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.32 : GPU-Accelerated Landmark Detection
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size, and NEVER bypasses credentials/registration. The real landmark datasets
# require registration, so this script only prints where to get them and how
# they map onto our loader, and defers to scripts/make_synthetic.py for the
# offline stand-in the demo uses.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.32 -- GPU-Accelerated Landmark Detection"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project decodes landmark HEATMAPS (a network's output tensors)."
echo "Real annotated landmark datasets require registration / a data-use"
echo "agreement, so we do NOT auto-download them. Sources (register first):"
echo "  * VerSe vertebral challenge  https://github.com/anjany/verse"
echo "        374 CT scans, 26 vertebral landmarks each."
echo "  * RSNA Vertebral Fracture Detection"
echo "        https://rsna-vertebral-labeling-level-detection.grand-challenge.org/"
echo "  * CephaloNet cephalometric landmark dataset (2D)."
echo "  * MICCAI 2015 prostate challenge landmark dataset."
echo
echo "To turn a real volume + a network's prediction into our input format,"
echo "export each landmark's heatmap tensor [Z,Y,X] to the layout documented"
echo "in data/README.md (nx ny nz L, then per-landmark: cx cy cz + voxels)."
echo
echo "The committed tiny sample in data/sample/ runs the demo offline. For a"
echo "larger SYNTHETIC problem (no registration needed), run:"
echo "    python scripts/make_synthetic.py --nx 64 --ny 64 --nz 64 --landmarks 26"
