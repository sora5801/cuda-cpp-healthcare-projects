#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 4.3 : MRI Reconstruction with Compressed Sensing
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URLs, and NEVER
# bypasses credentials/registration. Every real raw-k-space MRI dataset for this
# project sits behind a data-use agreement, so this script only PRINTS the
# registration instructions and links, and points at make_synthetic.py for an
# offline stand-in. The committed data/sample/ already lets the demo run offline.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 4.3 -- MRI Reconstruction with Compressed Sensing"
echo "[download_data] Target data dir: $DATA_DIR"
echo
echo "All real raw-k-space MRI datasets require a data-use agreement and CANNOT be"
echo "auto-downloaded. Register with the provider, then export one slice's k-space"
echo "into the text layout documented in data/README.md."
echo
echo "  fastMRI (NYU/Meta) -- knee + brain raw k-space (data-use agreement required):"
echo "     https://fastmri.med.nyu.edu/"
echo "     https://github.com/facebookresearch/fastMRI"
echo
echo "  Calgary-Campinas-359 -- multi-channel brain MRI k-space:"
echo "     https://sites.google.com/view/calgary-campinas-dataset/"
echo
echo "  SKM-TEA (Stanford knee MRI):"
echo "     https://github.com/StanfordMIMI/skm-tea"
echo
echo "The committed tiny SYNTHETIC sample in data/sample/ is enough to run the demo."
echo "For a larger synthetic problem, run:"
echo "    python scripts/make_synthetic.py --n 64 --keep 0.30 --iters 80"
echo
echo "When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip the download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY (never bypass)"
