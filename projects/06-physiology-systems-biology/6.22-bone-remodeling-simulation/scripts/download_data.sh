#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 6.22 : Bone Remodeling Simulation   (REDUCED-SCOPE teaching version)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints the source URLs, and
# NEVER bypasses credentials/registration. This teaching model runs entirely on
# the committed SYNTHETIC parameter file (data/sample/bone_params.txt) plus
# scripts/make_synthetic.py, so there is nothing to download to run the demo.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 6.22 -- Bone Remodeling Simulation"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "This project needs NO download: the committed synthetic sample at"
echo "  data/sample/bone_params.txt"
echo "is sufficient to build and run the demo offline. For a larger SYNTHETIC"
echo "problem, run:"
echo "    python scripts/make_synthetic.py --nx 64 --ny 48 --load-x0 28 --load-x1 35"
echo
echo "Real bone-imaging datasets you could adapt a voxel-FEM pipeline to"
echo "(segment a microCT stack into a bone/marrow voxel mask, then remodel):"
echo "  * OsteoArthritis Initiative (OAI): https://nda.nih.gov/oai/  (registration required)"
echo "  * PhysioNet bone datasets:         https://physionet.org     (credentialed use for some)"
echo "  * BoneJ morphometric examples:     https://bonej.org"
echo "  * MICCAI bone segmentation:        https://grand-challenge.org"
echo
echo "Respect every license; NEVER bypass registration. If redistribution is"
echo "forbidden, keep using the synthetic sample (see data/README.md)."
