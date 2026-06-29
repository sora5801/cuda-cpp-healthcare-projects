#!/usr/bin/env bash
# ===========================================================================
# scripts/download_data.sh  --  Fetch the FULL dataset (Linux / macOS)
# ---------------------------------------------------------------------------
# Project 10.14 -- LVAD / Rotary Blood Pump CFD & Hemolysis Prediction   (template skeleton)
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

echo "[download_data] Project 10.14 -- LVAD / Rotary Blood Pump CFD & Hemolysis Prediction"
echo "[download_data] Target data dir: $DATA_DIR"
echo

# TODO(impl): fill in the real dataset fetch. Template only prints guidance.
echo "TODO(impl): no full dataset wired up yet for this template skeleton."
echo "  Catalog dataset notes:"
echo "    FDA Benchmark Pump Dataset — PIV-measured flow in centrifugal/axial blood pumps (https://www.fda.gov/science-research/about-science-research-fda/computational-modeling-biomedical-devices); Multi-GPU IB Hemodynamics Benchmark (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7402620/); LVAD Thrombosis Simulation Archive (see https://arxiv.org/abs/2312.04761); HeartMate 3 geometry (anonymized, verify via Frontiers Cardiovasc Med)."
echo
echo "  The committed tiny sample in data/sample/ is enough to run the demo."
echo "  For a larger SYNTHETIC problem, run:"
echo "    python scripts/make_synthetic.py --n 1048576"
echo
echo "  When wiring a real dataset, follow this idempotent pattern:"
echo "    1) skip download if the file already exists with the right checksum"
echo "    2) print source URL + expected size + SHA256"
echo "    3) for credentialed sets, print registration instructions ONLY"
