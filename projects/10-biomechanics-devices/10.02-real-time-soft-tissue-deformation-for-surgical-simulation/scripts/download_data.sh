#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 10.2 -- Real-Time Soft-Tissue Deformation for Surgical Simulation   (template skeleton)
#
# CONTRACT (CLAUDE.md §8): idempotent, documented, prints source URL + expected
# size + checksum, and NEVER bypasses credentials/registration. Defers to
# scripts/make_synthetic.py for an offline stand-in when needed.
#
# Usage:  ./scripts/download_data.sh
# ===========================================================================
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "[download_data] Project 10.2 -- Real-Time Soft-Tissue Deformation for Surgical Simulation"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# TODO(impl): fill in the real dataset fetch. Template only prints guidance.
echo "TODO(impl): no full dataset wired up yet for this template skeleton."
echo "  Catalog dataset notes:"
echo "    SOFA Framework benchmark scenes — laparoscopic and open-surgery deformable organ models (https://www.sofa-framework.org/); Kaggle Liver CT Segmentation — 3D liver meshes for deformation benchmarking (https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation); MRI Breast Tissue Segmentation (nnU-Net preprocessed) for biomechanical modeling (https://arxiv.org/abs/2411.18784); iMSTK Test Suite — pre-built surgical scenario meshes (https://www.imstk.org/)."
echo
echo "  The committed tiny sample in data/sample/ is enough to run the demo."
echo "  For a larger SYNTHETIC problem, run:"
echo "    python scripts/make_synthetic.py --n 1048576"
echo
echo "  When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
