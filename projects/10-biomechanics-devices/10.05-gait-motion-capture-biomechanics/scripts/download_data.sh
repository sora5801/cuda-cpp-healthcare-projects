#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 10.5 -- Gait & Motion-Capture Biomechanics   (template skeleton)
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

echo "[download_data] Project 10.5 -- Gait & Motion-Capture Biomechanics"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# TODO(impl): fill in the real dataset fetch. Template only prints guidance.
echo "TODO(impl): no full dataset wired up yet for this template skeleton."
echo "  Catalog dataset notes:"
echo "    GaitRec — 2,084 patient bilateral ground reaction force (GRF) walking trials + 211 healthy controls (https://www.nature.com/articles/s41597-020-0481-z); CMU Motion Capture Database — 2500+ mocap sequences across diverse activities (http://mocap.cs.cmu.edu/); PhysioNet Gait/Posture Database — multi-camera + 17-IMU multimodal gait (https://physionet.org/content/multi-gait-posture/1.0.0/); Gait120 — comprehensive EMG + kinematic dataset (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12177048/)."
echo
echo "  The committed tiny sample in data/sample/ is enough to run the demo."
echo "  For a larger SYNTHETIC problem, run:"
echo "    python scripts/make_synthetic.py --n 1048576"
echo
echo "  When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
